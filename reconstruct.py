# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
from einops import repeat
from torchvision.utils import save_image
from torcheval.metrics.functional.aggregation.auc import auc
from diffusion import create_diffusion

from models import DiT_models
import argparse
import os

from tqdm import tqdm, trange

from train_seg import basic_config, model_config, data_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def reconstruct_show_process(model, dataset, output_folder, vae, device, batch_size, image_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=100)
    model.eval()  # important! This disables randomized embedding dropout

    indices = list(range(diffusion.num_timesteps))[::-1]
    show_indices = [indices[i] for i in range(0, 100, 10)]

    with torch.no_grad():
        t0 = torch.LongTensor([99]).to(device)
        for e in range(10):
            x, y = dataset[20 + e]
            x = x.unsqueeze(0).to(device)
            out_image_progress = [x]
            x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
            x_noised = diffusion.q_sample(x_latent, t0)
            for i in tqdm(indices):
                t = torch.tensor([i], device=device)
                pred_latent = diffusion.p_sample(model, x_latent, x_noised, t, w=0.)
                pred = vae.decode(pred_latent["sample"] / 0.18215).sample
                if i in show_indices:
                    out_image_progress.append(pred)
                x_noised = pred_latent["sample"]

            image_compare = torch.concat(out_image_progress, dim=3)
            save_image(image_compare, os.path.join(output_folder, f"prediction_guide{e}.png"), nrow=4, normalize=True,
                       value_range=(-1, 1))


def reconstruct(model, loader, output_folder, vae, device, batch_size, image_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=100)
    model.eval()  # important! This disables randomized embedding dropout
    pbar = tqdm(enumerate(loader), desc="Eval:")
    t = torch.LongTensor([20 for _ in range(batch_size)]).to(device)
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            # save_image(x, os.path.join(output_folder,f"origin_batch{i}.png"), nrow=4, normalize=True, value_range=(-1,1))
            x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
            x_noised = diffusion.q_sample(x_latent, t)
            # z = torch.randn_like(x_noised, device=device)
            pred_latent = diffusion.ddim_sample_loop(model, x_latent, x_noised.shape, noise=x_noised, eta=0.2)
            pred = vae.decode(pred_latent / 0.18215).sample
            # pred_direct = diffusion.p_sample(model, x_latent, x_noised, t)
            # pred = vae.decode(pred_direct["sample"] / 0.18215).sample
            # z = vae.decode(z / 0.18215).sample
            # print(torch.sum(x_noised-x_latent))
            image_compare = torch.concat((x, pred, torch.abs(x - pred)), dim=2)
            save_image(image_compare, os.path.join(output_folder, f"pred_batch{i}.png"), nrow=4, normalize=True,
                       value_range=(-1, 1))


def segmentation(recon_model, seg_model, loader, output_folder, vae, device, batch_size, image_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ratio = 1
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=200)
    seg_model.eval()  # important! This disables randomized embedding dropout
    pbar = tqdm(enumerate(loader), desc="Eval:")
    t = torch.LongTensor([20 for _ in range(batch_size)]).to(device)
    loss_list = []
    y_list = []
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            # y = y.to(device)
            latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            noised_x = diffusion.q_sample(latent_x, t)
            pred_latent_x = diffusion.ddim_sample_loop(recon_model, latent_x, noised_x.shape, noise=noised_x)

            pred_x = vae.decode((ratio * pred_latent_x + (1 - ratio) * latent_x) / 0.18215).sample
            # pred_x = vae.decode(pred_latent_x / 0.18215).sample
            seg_input = torch.cat((x, pred_x), dim=1)
            attr_loss, pred_y = seg_model(seg_input)
            recon_loss = torch.sum(torch.abs(x - pred_x), dim=(1, 2, 3))
            # pred_y[torch.where(pred_y >= 0.5)] = 1
            # pred_y[torch.where(pred_y < 0.5)] = 0
            mask_loss = torch.sum(pred_y[:, 0], dim=(1, 2))
            loss_list += (mask_loss).tolist()
            # recon_loss = torch.mean((pred_x-x)**2, dim=(1,2,3))
            # loss_list += (attr_loss+mask_loss+recon_loss).tolist()

            y_list += y.tolist()

            image_compare = torch.concat((x, pred_x, repeat(pred_y[:, 0], "b h w -> b c h w", c=3)), dim=2)
            save_image(image_compare, os.path.join(output_folder, f"pred_mask{i}.png"), nrow=8, normalize=True,
                       value_range=(-1, 1))
    result_auc = cal_auc(loss_list, y_list)
    print(result_auc)


def cal_auc(pred, y):
    p = sum(y)
    n = len(y) - sum(y)
    tmp_p = 0
    tpr = [0]
    fpr = [0]
    sorted_pred = sorted(pred)
    for i in range(len(sorted_pred)):
        j = pred.index(sorted_pred[i])
        if y[j] == 0:
            tmp_p += 1
            tpr.append(tmp_p / p)
            fpr.append(fpr[-1])
        else:
            tpr.append(tpr[-1])
            fpr.append((i - tmp_p + 1) / n)
    tpr = torch.tensor(tpr)
    fpr = torch.tensor(fpr)
    return auc(fpr, tpr)


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    checkpoint_dir, logger, device = basic_config(args)
    test_loader = data_config(args, logger, training=False)
    diffusion, vae, recon_model, seg_model, seg_opt, scheduler, seg_loss_func = model_config(args, device, logger)

    # reconstruct(recon_model, loader, args.output_folder, vae, device, batch_size=args.batch_size)
    segmentation(recon_model, seg_model, test_loader, args.output_folder, vae, device, batch_size=args.batch_size)
    # reconstruct_show_process(recon_model, test_dataset, args.output_folder, vae, device, batch_size=args.batch_size)
    # Save and display images:
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=str,
                        default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\test")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-L/4")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recon-ckpt", type=str, default="results/DiT-L-4-cable/checkpoints/recon.pt",
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--seg-ckpt", type=str, default="results/DiT-L-4-cable/checkpoints/seg.pt")
    parser.add_argument("--output-folder", type=str, default=r"samples/eval")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
