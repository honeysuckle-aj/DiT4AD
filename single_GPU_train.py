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
from models import DiT_models
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from ad_dataset import MaskedDataset, TestDataset, load_textures, pair
from reconstruct import reconstruct

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
    model = DiT_models[args.model](
        input_size=latent_size,
        # num_classes=args.num_classes
    )
    # use pre-trained model
    if args.pre_trained != "":
        state_dict = find_model(args.pre_trained)
        model.load_state_dict(state_dict)

    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    model = model.to(device)
    requires_grad(ema, False)
    # model = DDP(model.to(device), device_ids=[rank])  # parallel computing
    diffusion = create_diffusion(timestep_respacing="",
                                 diffusion_steps=100)  # default: 1000 steps, linear noise schedule. in training use ddpm config
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.98)

    # Setup data:
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])

    textures = load_textures(args.texture_path, image_size=pair(args.image_size))

    # dataset = ImageFolder(args.data_path, transform=transform)
    dataset = MaskedDataset(args.data_path, textures=textures)
    test_set = TestDataset(args.test_set)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
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
    test_loader = DataLoader(test_set, batch_size=8, drop_last=True)
    logger.info(f"Training Dataset contains {len(dataset)} images")
    logger.info(f"Eval Dataset contains {len(test_set)} images")
    #  before training, test reconstruct
    reconstruct(model, test_loader, args.output_folder, vae, device, batch_size=8)
    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    start_time = time()
    # sum_loss = 0

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch_batch in range(args.epochs // args.log_every_epoch):
        logger.info(f"Beginning epoch batch {epoch_batch}...")
        p_bar = tqdm(range(args.log_every_epoch), desc=f"Training {epoch_batch} th epoch batch", unit="epoch")
        running_loss = 0
        for epoch in p_bar:
            epoch_loss = 0
            # sampler.set_epoch(epoch)
            t_mask = 10  # in this 100 steps, the model is trained to reconstruct the origin images from the masked images
            mask_epoch = 10  # masked images will be trained once every 10 epochs
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
                if i % mask_epoch == mask_epoch - 1:
                    # train masked images
                    t = torch.randint(diffusion.num_timesteps - t_mask, diffusion.num_timesteps, (img.shape[0],),
                                      device=device)
                    loss_dict = diffusion.training_losses(model, img, mask_img, t, sum_steps=diffusion.num_timesteps)
                else:
                    # train normal images
                    t = torch.randint(0, diffusion.num_timesteps, (img.shape[0],), device=device)
                    loss_dict = diffusion.training_losses(model, img, img, t, sum_steps=diffusion.num_timesteps)
                # t = repeat(torch.randint(0, diffusion.num_timesteps, (1,), device=device),"l -> b", b=img.shape[0]) # img.shape[0] -> batch size
                # TODO
                # model_kwargs = dict(y=y)

                loss = loss_dict["loss"].mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                update_ema(ema, model)

                # Log loss values:
                running_loss += loss.item()
                epoch_loss += loss.item()
            p_bar.set_postfix(loss=epoch_loss)

        end_time = time()
        steps_per_sec = args.log_every_epoch * args.batch_size / (end_time - start_time)
        # Reduce loss history over all processes:
        avg_loss = torch.tensor(running_loss / (args.log_every_epoch * args.batch_size), device=device)
        # dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        # avg_loss = avg_loss.item() / dist.get_world_size()
        avg_loss = avg_loss.item()
        logger.info(
            f"(epoch batch={epoch_batch:05d}) Average Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
        # Reset monitoring variables:
        start_time = time()

        # Save DiT checkpoint:
        if epoch_batch % args.ckpt_every_epoch == args.ckpt_every_epoch - 1:
            # if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "args": args
            }
            checkpoint_path = f"{checkpoint_dir}/last.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            # dist.barrier()
            # p_bar.set_postfix(loss=sum_loss, train_step=train_steps)
            # logger.info(f"(epoch={epoch:07d}) Train Loss: {sum_loss:.4f}")
            # torch.cuda.empty_cache()
            reconstruct(model, test_loader, args.output_folder, vae, device, batch_size=8)
            model.train()

    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Training Done!")
    torch.cuda.empty_cache()
    reconstruct(model, test_loader, args.output_folder, vae, device, batch_size=8)
    # cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\capsule\train\good")
    parser.add_argument("--test-set", type=str,
                        default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\capsule\test")
    parser.add_argument("--texture-path", type=str, default="dataset/textures")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/4")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=6000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every-epoch", type=int, default=100)
    parser.add_argument("--ckpt-every-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-folder", type=str, default="samples/mask_capsule")
    parser.add_argument("--pre-trained", type=str, default="")
    args = parser.parse_args()
    main(args)
