# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
from copy import deepcopy

import PIL.Image
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import repeat, rearrange
from torchvision.utils import save_image
# from torcheval.metrics.functional.aggregation.auc import auc
from sklearn.metrics import auc, roc_curve, average_precision_score, roc_auc_score
from diffusion import create_diffusion

from models import DiT_models
import argparse
import os

from tqdm import tqdm, trange

from configs import basic_config, model_config, data_config

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
    pbar = tqdm(enumerate(loader), desc="Reconstruct:")
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
            # image_compare = torch.concat((x, pred, torch.abs(x - pred)), dim=2)
            save_image(pred,
                       f"E:/DataSets/AnomalyDetection/mvtec_anomaly_detection/cable/recon/train/good/recon/train_{i}.png",
                       normalize=True)


def segmentation(recon_model, seg_model, loader, output_folder, vae, device, batch_size, image_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ratio = 1
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=200)
    seg_model.eval()  # important! This disables randomized embedding dropout
    pbar = tqdm(enumerate(loader), desc="Eval:")
    t = torch.LongTensor([20 for _ in range(batch_size)]).to(device)
    loss_list = []
    gt_mask_list = np.array([])
    pred_mask_list = np.array([])
    with torch.no_grad():
        for i, (x, y) in pbar:
            x = x.to(device)
            y = y.to(device)
            latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            noised_x = diffusion.q_sample(latent_x, t)
            pred_latent_x = diffusion.ddim_sample_loop(recon_model, latent_x, noised_x.shape, noise=noised_x)

            pred_x = vae.decode((ratio * pred_latent_x + (1 - ratio) * latent_x) / 0.18215).sample
            # pred_x = vae.decode(pred_latent_x / 0.18215).sample
            seg_input = torch.cat((x, pred_x), dim=1)
            feat_loss, pred_y = seg_model(seg_input)
            gt_mask_list = np.append(gt_mask_list, y.flatten().detach().cpu().numpy())
            pred_mask_list = np.append(pred_mask_list, pred_y[:, 0].flatten().detach().cpu().numpy())
            recon_loss = torch.sum(torch.abs(x - pred_x), dim=(1, 2, 3))
            pred_y[torch.where(pred_y >= 0.5)] = 1
            pred_y[torch.where(pred_y < 0.5)] = 0
            seg_loss = torch.sum(pred_y[:, 0], dim=(1, 2))
            loss_list += (1 / (recon_loss * 0.070 + feat_loss * 0.00225 + seg_loss * 5.500)).tolist()
            # output_list[0] += recon_loss.tolist()
            # output_list[1] += feat_loss.tolist()
            # output_list[2] += seg_loss.tolist()
            # output_list[3] += y.tolist()
            # recon_loss = torch.mean((pred_x-x)**2, dim=(1,2,3))
            # loss_list += (attr_loss+mask_loss+recon_loss).tolist()

            # y_list += y.tolist()
            # save_image(pred_x, os.path.join(output_folder, f"recon/{i}.png"), normalize=True)
            # save_image(x, os.path.join(output_folder, f"origin/{i}.png"), normalize=True)
            # yi = y[0].detach().numpy()
            # yi = rearrange(np.repeat(yi, 3, axis=0), "c h w -> h w c")
            # gt_mask = PIL.Image.fromarray(yi)
            # gt_mask.save(os.path.join(output_folder, f"gt_mask/{i}.png"))
            # save_image(repeat(y[0], "b h w -> b c h w", c=3), os.path.join(output_folder, f"gt_mask/{i}.png"))
            # save_image(repeat(pred_y[:, 0], "b h w -> b c h w", c=3), os.path.join(output_folder, f"mask/{i}.png"), normalize=True)
            # image_compare = torch.concat((x, pred_x, repeat(pred_y[:, 0], "b h w -> b c h w", c=3)), dim=2)
            # save_image(image_compare, os.path.join(output_folder, f"pred_mask{i}.png"), nrow=8, normalize=True,
            #            value_range=(-1, 1))
            # save_image(image_compare, os.path.join(output_folder, f"pred_paper{i}.png"), nrow=args.batch_size, normalize=True)
    # result_auc = cal_auc(loss_list, y_list)
    # print(result_auc)
    auroc_pixel = round(roc_auc_score(gt_mask_list, pred_mask_list), 3) * 100
    print("Pixel AUC-ROC:", auroc_pixel)
    #
    ap_pixel = round(average_precision_score(gt_mask_list, pred_mask_list), 3) * 100
    print("Pixel-AP:", ap_pixel)
    # output_list = np.array(output_list)
    # np.savetxt(os.path.join(output_folder, "output.txt"), output_list, fmt="%.5f")
    # print("output")


def segmentation_cal_ap(recon_model, seg_model, loader, output_folder, vae, device, batch_size, image_size=256):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    ratio = 0.9
    diffusion = create_diffusion(timestep_respacing="", diffusion_steps=200)
    seg_model.eval()  # important! This disables randomized embedding dropout
    pbar = tqdm(enumerate(loader), desc="Eval:")
    t = torch.LongTensor([20 for _ in range(batch_size)]).to(device)
    # y_list = []
    pred_mask_list = []
    gt_mask_list = []
    with torch.no_grad():
        for i, (x, mask) in pbar:
            x = x.to(device)
            mask = mask.to(device)
            latent_x = vae.encode(x).latent_dist.sample().mul_(0.18215)
            noised_x = diffusion.q_sample(latent_x, t)
            pred_latent_x = diffusion.ddim_sample_loop(recon_model, latent_x, noised_x.shape, noise=noised_x)

            pred_x = vae.decode((ratio * pred_latent_x + (1 - ratio) * latent_x) / 0.18215).sample
            # pred_x = vae.decode(pred_latent_x / 0.18215).sample
            seg_input = torch.cat((x, pred_x), dim=1)
            feat_loss, pred_y = seg_model(seg_input)
            gt_mask_list.append(mask[:, 0])
            pred_mask_list.append(pred_y[:, 0])
            save_image(x, f"{output_folder}/{i}-x.png", normalize=True)
            save_image(torch.cat((pred_y[:, 0], mask[:, 0]), dim=2), f"{output_folder}/{i}-th.png")

    ap = cal_ap(pred_mask_list, gt_mask_list)
    print(ap)

    # output_list = np.array(output_list)
    # np.savetxt(os.path.join(output_folder, "output.txt"), output_list, fmt="%.5f")
    # print("output")


def cal_auc(pred, y):
    p = sum(y)
    n = len(y) - p
    tmp_tp = 0
    tmp_fp = 0
    tpr = [0]
    fpr = [0]
    sorted_pred = list(zip(pred, y))
    sorted_pred = sorted(sorted_pred, key=lambda x: x[0])
    for i in range(len(sorted_pred)):
        if sorted_pred[i][1] == 1:
            tmp_tp += 1
            tpr.append(tmp_tp / p)
            fpr.append(fpr[-1])
        else:
            tmp_fp += 1
            tpr.append(tpr[-1])
            fpr.append(tmp_fp / n)
    tpr = torch.tensor(tpr)
    fpr = torch.tensor(fpr)
    return auc(fpr, tpr)


def cal_ap(pred, y):
    precision = [1]
    recall = [0]
    for i in range(0, 101):
        threshold = (100 - i) / 100
        p, r = cal_pr(pred, y, threshold)
        precision.append(p)
        recall.append(r)
    precision = torch.tensor(precision)
    recall = torch.tensor(recall)
    plt.plot(recall, precision)
    plt.show()
    return auc(recall, precision)


def cal_pr(pred: list, y: list, threshold: float):
    T = 0  # ground truth
    P = 0  # pred
    TP = 0
    for i in range(len(pred)):
        temp_y = y[i]
        T_points = torch.where(temp_y > 0)
        temp_pred = pred[i].clone()
        temp_pred[torch.where(temp_pred >= threshold)] = 1
        temp_pred[torch.where(temp_pred < threshold)] = 0
        P_points = torch.where(temp_pred == 1)
        T += len(T_points[0])
        P += len(P_points[0])

        inter = temp_pred * temp_y
        TP += len(torch.where(inter > 0)[0])
    return TP / P, TP / T  # precision, recall


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    checkpoint_dir, logger, device = basic_config(args)
    test_loader = data_config(args, logger, training=False, use_mask=True)
    diffusion, vae, recon_model, seg_model, seg_opt, scheduler, seg_loss_func = model_config(args, device, logger)

    # reconstruct(recon_model, test_loader, args.output_folder, vae, device, batch_size=args.batch_size)
    # segmentation_cal_ap(recon_model, seg_model, test_loader, args.output_folder, vae, device,
    #                     batch_size=args.batch_size)
    segmentation(recon_model, seg_model, test_loader, args.output_folder, vae, device,
                 batch_size=args.batch_size)
    # reconstruct_show_process(recon_model, test_dataset, args.output_folder, vae, device, batch_size=args.batch_size)
    # Save and display images:
    # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--test-set", type=str,
    #                     default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\test")
    parser.add_argument("--test-set", type=str,
                        default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\test")
    parser.add_argument("--mask-set", type=str,
                        default=r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\ground_truth\bad")
    parser.add_argument("--batch-size", type=int, default=1)
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
    parser.add_argument("--output-folder", type=str, default=r"samples/paper")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()
    main(args)
