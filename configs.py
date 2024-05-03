import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import torch
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader

from ad_dataset import load_textures, MaskedDataset, SegTrainDataset, TestDataset, pair
from diffusion import create_diffusion
from download import find_model
from models import SegCNN, DiT_models


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


# def cleanup():
#     """
#     End DDP training.
#     """
#     dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """

    logging.basicConfig(
        level=logging.INFO,
        format='[ [34m%(asctime)s [0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def basic_config(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = "cuda:0" if torch.cuda.is_available() else "CPU"

    torch.manual_seed(1)
    torch.cuda.set_device(device)
    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    return checkpoint_dir, logger, device


def data_config(args, logger, training=True, use_cache=False, use_mask=False):

    test_set = TestDataset(args.test_set)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, drop_last=True)
    logger.info(f"Eval Dataset contains {len(test_set)} images")
    if training:
        textures = load_textures(args.texture_path, image_size=pair(args.image_size))
        if not use_cache:
            dataset = MaskedDataset(args.data_path, textures=textures)
        else:
            dataset = SegTrainDataset(args.data_path)
        loader = DataLoader(
            dataset,
            # batch_size=int(args.global_batch_size // dist.get_world_size()),
            batch_size=args.batch_size,
            shuffle=False,
            # sampler=sampler,
            num_workers=args.num_workers,
            # pin_memory=True,
            drop_last=True
        )
        logger.info(f"Training Dataset contains {len(dataset)} images")

        return loader, test_loader
    else:
        return test_loader


def model_config(args, device, logger):
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

    seg_model = SegCNN(in_channels=3, out_channels=2)
    if args.seg_ckpt != "":
        seg_checkpoint = find_model(args.seg_ckpt)
        seg_model.load_state_dict(seg_checkpoint["model"])

    seg_model = seg_model.to(device)
    seg_model.train()

    seg_opt = torch.optim.AdamW(seg_model.parameters(), lr=8e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=seg_opt, T_max=10, eta_min=1e-6)

    diffusion = create_diffusion(timestep_respacing="",
                                 diffusion_steps=100)  # default: 1000 steps, linear noise schedule. in training use ddpm config
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"SegCNN Parameters: {sum(p.numel() for p in seg_model.parameters()):,}")

    seg_loss_func = torch.nn.BCELoss(reduction='sum')
    # logger.info(f"Training for {args.epochs} epochs...")

    recon_model = DiT_models[args.model](
        input_size=latent_size,
        # num_classes=args.num_classes
    )
    recon_checkpoint = find_model(args.recon_ckpt)
    recon_model.load_state_dict(recon_checkpoint["ema"])
    ema = deepcopy(recon_model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    recon_model = recon_model.to(device)
    recon_model.eval()
    logger.info(f"DiT Parameters: {sum(p.numel() for p in recon_model.parameters()):,}, ")
    return diffusion, vae, recon_model, seg_model, seg_opt, scheduler, seg_loss_func
