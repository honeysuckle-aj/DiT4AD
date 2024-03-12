# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
from tqdm import tqdm
import argparse
import logging
import os

from download import find_model
from models import DiT_models, ViT
from experiment.VIT import train_seg_loss
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from ad_dataset import MaskedDataset, TestDataset, load_textures, pair
from reconstruct import reconstruct, segmentation

# the first flag below was False when we tested this script but True makes A100 training a lot faster:


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

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
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert args.epochs % args.log_every_epoch == 0, "epochs must be divisible by log-every-epoch"

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    device = "cuda:0" if torch.cuda.is_available() else "CPU"

    torch.manual_seed(1)
    torch.cuda.set_device(device)
    # Setup an experiment folder:

    os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    # logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    recon_model = DiT_models[args.model](
        input_size=latent_size,
        # num_classes=args.num_classes
    )
    checkpoint = find_model(args.DiT_model)
    recon_model.load_state_dict(checkpoint["ema"])
    seg_model = ViT(input_size=256, patch_size=16, output_size=256, hidden_size=768, depth=3, num_heads=6, mlp_ratio=4,
                    in_channels=9, dropout=0.1)
    # use pre-trained model
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    recon_model = recon_model.to(device)
    seg_model = seg_model.to(device)
    seg_opt = torch.optim.AdamW(seg_model.parameters(), lr=1e-5, weight_decay=0.99)

    diffusion = create_diffusion(timestep_respacing="ddim20",
                                 diffusion_steps=100)  # default: 1000 steps, linear noise schedule. in training use ddpm config
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in recon_model.parameters()):,}")

    textures = load_textures(args.texture_path, image_size=pair(args.image_size))

    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = MaskedDataset(args.data_path, textures=textures)
    test_set = TestDataset(args.test_set)
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
    test_loader = DataLoader(test_set, batch_size=args.batch_size, drop_last=True)
    logger.info(f"Training Dataset contains {len(dataset)} images")
    logger.info(f"Eval Dataset contains {len(test_set)} images")

    # before training, test reconstruction
    # reconstruct(model, test_loader, args.output_folder, vae, device, batch_size=8)

    # Prepare models for training:
    recon_model.eval()  # important! This enables embedding dropout for classifier-free guidance
    seg_model.train()


    # Variables for monitoring/logging purposes:
    start_time = time()
    # sum_loss = 0
    seg_loss_func = torch.nn.BCEWithLogitsLoss(reduction='sum')
    logger.info(f"Training for {args.epochs} epochs...")
    t = torch.LongTensor([len(diffusion.use_timesteps) - 1 for _ in range(args.batch_size)]).to(device)
    segmentation(recon_model, seg_model, test_loader, args.output_folder, vae, device, args.batch_size, image_size=256)
    for epoch_batch in range(args.epochs // args.log_every_epoch):
        logger.info(f"Beginning epoch batch {epoch_batch}...")
        p_bar = tqdm(range(args.log_every_epoch), desc=f"Training {epoch_batch} th epoch batch", unit="epoch")
        seg_batch_loss = 0
        for epoch in p_bar:
            recon_epoch_loss = 0
            seg_epoch_loss = 0
            # sampler.set_epoch(epoch)
            for i, (img, mask_img, mask) in enumerate(loader):
                img = img.to(device)
                mask_img = mask_img.to(device)
                mask = mask.to(device)
                # x = x.to(device)
                # y = y.to(device)
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    # x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    img = vae.encode(img).latent_dist.sample().mul_(0.18215)
                    mask_img = vae.encode(mask_img).latent_dist.sample().mul_(0.18215)
                    noised_img = diffusion.q_sample(mask_img, t)
                    # pred_xstart = diffusion.p_sample(recon_model, mask_img, noised_img, t)["pred_xstart"]
                    pred_xstart = diffusion.ddim_sample_loop(recon_model, mask_img, mask_img.shape, noised_img)

                # t = repeat(torch.randint(0, diffusion.num_timesteps, (1,), device=device),"l -> b", b=img.shape[0]) # img.shape[0] -> batch size
                # TODO
                # model_kwargs = dict(y=y)

                pred_y, seg_loss = train_seg_loss(seg_model, seg_loss_func, mask_img, pred_xstart, mask, vae)
                seg_opt.zero_grad()
                seg_loss.backward()
                seg_opt.step()

                # Log loss values:
                seg_batch_loss += seg_loss.item()
                seg_epoch_loss += seg_loss.item()
            p_bar.set_postfix(recon_loss=recon_epoch_loss, seg_loss=seg_epoch_loss)

        end_time = time()
        steps_per_sec = args.log_every_epoch * args.batch_size / (end_time - start_time)
        seg_avg_loss = seg_batch_loss / (args.log_every_epoch * args.batch_size)
        logger.info(
            f"(epoch batch={epoch_batch:05d}), Segmentation Loss: {seg_avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
        # Reset monitoring variables:
        start_time = time()

        # Save DiT checkpoint:
        if epoch_batch % args.ckpt_every_epoch == args.ckpt_every_epoch - 1:
            # if rank == 0:
            seg_checkpoint = {
                "model": seg_model.state_dict(),
                "opt": seg_opt.state_dict(),
            }
            checkpoint_path = checkpoint_dir
            torch.save(seg_checkpoint, f"{checkpoint_path}/seg.pt")
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            segmentation(recon_model, seg_model, test_loader, args.output_folder, vae, device, args.batch_size, image_size=256)
            # recon_model.train()
            seg_model.train()

    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Training Done!")
    torch.cuda.empty_cache()
    # cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\train\good")
    parser.add_argument("--test-set", type=str,
                        default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\test")
    parser.add_argument("--texture-path", type=str, default="dataset/textures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--log-every-epoch", type=int, default=100)
    parser.add_argument("--ckpt-every-epoch", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--output-folder", type=str, default="samples/mask_cable")
    parser.add_argument("--DiT-model", type=str, default="results/DiT-L-4-cable/checkpoints/last.pt")
    parser.add_argument("--pre-trained", type=str, default="")
    args = parser.parse_args()
    main(args)
