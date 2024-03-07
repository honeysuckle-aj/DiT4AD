# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
from ad_dataset import TestDataset, pair
import argparse
import os

from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def reconstruct(model, loader, output_folder, vae, device, batch_size, image_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=100)
    model.eval()  # important! This disables randomized embedding dropout
    pbar = tqdm(enumerate(loader), desc="Eval:")
    t = torch.LongTensor([len(diffusion.use_timesteps) - 1 for _ in range(batch_size)]).to(device)
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            # save_image(x, os.path.join(output_folder,f"origin_batch{i}.png"), nrow=4, normalize=True, value_range=(-1,1))
            x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
            x_noised = diffusion.q_sample(x_latent, t)
            # z = torch.randn_like(x_noised, device=device)
            pred_latent = diffusion.p_sample_loop(model, x_latent, x_noised.shape, noise=x_noised)
            pred = vae.decode(pred_latent / 0.18215).sample
            # z = vae.decode(z / 0.18215).sample
            # print(torch.sum(x_noised-x_latent))
            image_compare = torch.concat((x, pred), dim=2)
            save_image(image_compare, os.path.join(output_folder, f"pred_batch{i}.png"), nrow=4, normalize=True,
                       value_range=(-1, 1))


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    # Load test dataset
    test_dataset = TestDataset(args.dataset)
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        # num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]

    # # Create sampling noise:
    # n = len(class_labels)
    # z = torch.randn(n, 4, latent_size, latent_size, device=device)
    # y = torch.tensor(class_labels, device=device)

    # # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * n, device=device)
    # y = torch.cat([y, y_null], 0)
    # model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sample images:
    # samples = diffusion.p_sample_loop(
    #     model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    # )
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    # samples = vae.decode(samples / 0.18215).sample

    reconstruct(model, loader, args.output_folder, vae, device, batch_size=args.batch_size)
    # Save and display images:
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\capsule\test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/4")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default="results/008-DiT-L-4/checkpoints/last.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--output-folder", type=str, default=r"samples")
    args = parser.parse_args()
    main(args)
