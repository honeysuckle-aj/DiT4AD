# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch

from torchvision.utils import save_image
from tqdm import tqdm
import argparse
import os

from configs import basic_config, data_config, model_config
from models import DiT_models, SegCNN

from reconstruct import segmentation

# the first flag below was False when we tested this script but True makes A100 training a lot faster:


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def make_data(args):
    checkpoint_dir, logger, device = basic_config(args)
    training_loader, test_loader = data_config(args, logger)
    diffusion, vae, recon_model, seg_model, seg_opt, scheduler, seg_loss_func = model_config(args, device, logger)
    t = torch.ones((args.batch_size,), dtype=torch.long, device=device) * 20
    t = t.to(device)
    train_folder = os.path.join(args.output_folder, "train")
    # test_folder = os.path.join(args.output_folder, "test")
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        os.makedirs(os.path.join(train_folder, "good", "recon"))
        os.makedirs(os.path.join(train_folder, "good", "origin"))
        os.makedirs(os.path.join(train_folder, "good", "mask"))
        # os.makedirs(os.path.join(test_folder, "good"))
        # os.makedirs(os.path.join(test_folder, "bad"))
    with torch.no_grad():
        for i, (img, mask_img, mask) in tqdm(enumerate(training_loader)):
            mask_img = mask_img.to(device)
            latent_img = vae.encode(mask_img).latent_dist.sample().mul_(0.18215)

            noised_img = diffusion.q_sample(x_start=latent_img, t=t)
            pred_xstart = diffusion.ddim_sample_loop(recon_model, noised_img, noised_img.shape,
                                                     noise=noised_img)
            pred_img = vae.decode(pred_xstart / 0.18215).sample
            pred_img = pred_img  # .view(256,256,3).detach().cpu().numpy()
            # img = Image.fromarray(pred_img)
            # img.save(f"{train_folder}/good/train_{i}.png")
            save_image(pred_img, f"{train_folder}/good/recon/train_{i}.png", normalize=True)
            save_image(mask_img, f"{train_folder}/good/origin/train_{i}.png", normalize=True)
            save_image(mask, f"{train_folder}/good/mask/train_{i}.png")


#################################################################################
#                                  Training Loop                                #
#################################################################################
def train_recon(args):
    checkpoint_dir, logger, device = basic_config(args)
    training_loader, test_loader = data_config(args, logger)
    diffusion, vae, recon_model, seg_model, seg_opt, scheduler, seg_loss_func = model_config(args, device, logger)
    t = torch.ones((args.batch_size,), dtype=torch.long, device=device) * 5
    t = t.to(device)
    # ratio = 0.8
    segmentation(recon_model, seg_model, test_loader, args.output_folder, vae, device, args.batch_size,
                 image_size=256)
    for epoch_batch in range(args.epochs // args.log_every_epoch):
        logger.info(f"Beginning epoch batch {epoch_batch}...")
        p_bar = tqdm(range(args.log_every_epoch), desc=f"Training {epoch_batch} th epoch batch", unit="epoch")
        seg_batch_loss = 0
        for epoch in p_bar:
            # recon_epoch_loss = 0
            seg_epoch_loss = 0
            for i, (img, mask_img, mask) in enumerate(training_loader):
                # img = img.to(device)
                mask_img = mask_img.to(device)
                mask = mask.to(device)
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    # x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    # img = vae.encode(img).latent_dist.sample().mul_(0.18215)
                    latent_img = vae.encode(mask_img).latent_dist.sample().mul_(0.18215)

                    noised_img = diffusion.q_sample(x_start=latent_img, t=t)
                    pred_xstart = diffusion.ddim_sample_loop(recon_model, noised_img, noised_img.shape,
                                                             noise=noised_img)
                    pred_img = vae.decode(pred_xstart / 0.18215).sample
                    # pred_xstart = ratio * pred_xstart + (1 - ratio) * mask_img

                pred_y, seg_loss = seg_model.train_loss(seg_loss_func, latent_img, pred_img, mask)
                seg_opt.zero_grad()
                seg_loss.backward()
                seg_opt.step()
                scheduler.step()
                # Log loss values:

                seg_batch_loss += seg_loss.item()
                seg_epoch_loss += seg_loss.item()

                p_bar.set_postfix(seg_loss=seg_epoch_loss)

        # end_time = time()
        # steps_per_sec = args.log_every_epoch * args.batch_size / (end_time - start_time)
        # recon_avg_loss = recon_batch_loss / (args.log_every_epoch * args.batch_size)
        seg_avg_loss = seg_batch_loss / (args.log_every_epoch * args.batch_size)
        logger.info(
            f"(epoch batch={epoch_batch:05d}),Seg Loss: {seg_avg_loss:.4f}")
        # Reset monitoring variables:
        # start_time = time()

        # Save DiT checkpoint:
        if epoch_batch % args.ckpt_every_epoch == args.ckpt_every_epoch - 1:
            seg_checkpoint = {
                "model": seg_model.state_dict(),
                "opt": seg_opt.state_dict(),
            }
            checkpoint_path = checkpoint_dir
            # torch.save(recon_checkpoint, f"{checkpoint_path}/recon.pt")
            torch.save(seg_checkpoint, f"{checkpoint_path}/seg.pt")
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            segmentation(recon_model, seg_model, test_loader, args.output_folder, vae, device, args.batch_size,
                         image_size=256)
            # recon_model.train()
            seg_model.train()
        logger.info("Training Done!")
        torch.cuda.empty_cache()
        # cleanup()


def train_aug(args):
    checkpoint_dir, logger, device = basic_config(args)
    training_loader, test_loader = data_config(args, logger, use_cache=True)
    diffusion, vae, recon_model, seg_model, seg_opt, scheduler, seg_loss_func = model_config(args, device, logger)
    segmentation(recon_model, seg_model, test_loader, args.output_folder, vae, device, args.batch_size,
                 image_size=256)
    for epoch_batch in range(args.epochs // args.log_every_epoch):
        logger.info(f"Beginning epoch batch {epoch_batch}...")
        p_bar = tqdm(range(args.log_every_epoch), desc=f"Training {epoch_batch} th epoch batch", unit="epoch")
        seg_batch_loss = 0
        for epoch in p_bar:
            # recon_epoch_loss = 0
            seg_epoch_loss = 0
            for i, (img, recon_img, mask) in enumerate(training_loader):
                img = img.to(device)
                recon_img = recon_img.to(device)
                mask = mask.to(device)

                pred_y, seg_loss = seg_model.train_loss(seg_loss_func, img, recon_img, mask)
                seg_opt.zero_grad()
                seg_loss.backward()
                seg_opt.step()
                scheduler.step()
                # Log loss values:

                seg_batch_loss += seg_loss.item()
                seg_epoch_loss += seg_loss.item()

                p_bar.set_postfix(seg_loss=seg_epoch_loss)

        # end_time = time()
        # steps_per_sec = args.log_every_epoch * args.batch_size / (end_time - start_time)
        # recon_avg_loss = recon_batch_loss / (args.log_every_epoch * args.batch_size)
        seg_avg_loss = seg_batch_loss / (args.log_every_epoch * args.batch_size)
        logger.info(
            f"(epoch batch={epoch_batch:05d}),Seg Loss: {seg_avg_loss:.4f}")
        # Reset monitoring variables:
        # start_time = time()

        # Save DiT checkpoint:
        if epoch_batch % args.ckpt_every_epoch == args.ckpt_every_epoch - 1:
            seg_checkpoint = {
                "model": seg_model.state_dict(),
                "opt": seg_opt.state_dict(),
            }
            checkpoint_path = checkpoint_dir
            # torch.save(recon_checkpoint, f"{checkpoint_path}/recon.pt")
            torch.save(seg_checkpoint, f"{checkpoint_path}/seg.pt")
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            segmentation(recon_model, seg_model, test_loader, args.output_folder, vae, device, args.batch_size,
                         image_size=256)
            # recon_model.train()
            seg_model.train()
        logger.info("Training Done!")
        torch.cuda.empty_cache()
        # cleanup()


def main(args):
    """
    Trains a new DiT model.
    """
    # make_data(args)
    # train_recon(args)
    train_aug(args)
    # logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:

    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str,
                        default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\recon\train\good")
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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--log-every-epoch", type=int, default=100)
    parser.add_argument("--ckpt-every-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-folder", type=str,
                        default=r"samples/cable_seg")
    parser.add_argument("--recon-ckpt", type=str, default="results/DiT-L-4-cable/checkpoints/recon.pt")
    parser.add_argument("--seg-ckpt", type=str, default="")
    args = parser.parse_args()
    main(args)
