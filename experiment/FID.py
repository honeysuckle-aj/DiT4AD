import os

import numpy as np
import torch
from PIL import Image
from pytorch_fid import fid_score
from tqdm import tqdm


def cal_fid(real_image, gen_image):
    # eval_model = torchvision.models.inception_v3(pretrained=True)
    # transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    # ])
    device = "cpu"
    fid = fid_score.calculate_fid_given_paths([real_image, gen_image], batch_size=16, device=device, dims=64)
    return fid
    # print(fid)


def cut_bins(origin_path, output_folder, origin_size=256, bin_radius=32, num_row=7):
    assert (origin_size - bin_radius * 2) % (num_row - 1) == 0, "size incorrect"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = []

    for f in os.listdir(origin_path):
        fn = os.path.join(origin_path, f)
        im = Image.open(fn).resize((origin_size, origin_size))
        images.append(im)

    square_xy = []
    block_size = (origin_size - bin_radius * 2) // (num_row - 1)
    for xi in range(1, num_row + 1):
        for yi in range(1, num_row + 1):
            square_xy.append((xi * block_size - bin_radius, yi * block_size - bin_radius, xi * block_size + bin_radius,
                              yi * block_size + bin_radius))

    for i in tqdm(range(len(square_xy))):
        sub_folder = os.path.join(output_folder, f"{i}th_bin")
        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)
        box = square_xy[i]
        for j in range(len(images)):
            im = images[j].crop(box)
            fn = os.path.join(sub_folder, f"{j}.png")
            im.save(fn)

def cal_sfid(real_folder, gen_folder):
    real_image = sorted(os.listdir(real_folder))
    gen_image = sorted(os.listdir(gen_folder))
    assert len(real_image) == len(gen_image), "not correspond"

    fid_list = []
    for i in range(len(real_image)):

        r_im = os.path.join(real_folder, real_image[i])
        g_im = os.path.join(gen_folder, gen_image[i])
        fid = cal_fid(r_im, g_im)
        fid_list.append(fid)
        print(i, fid)

    return np.mean(np.array(fid_list))

if __name__ == "__main__":

    # image_path = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\recon\train\good\recon"
    # output_folder = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\sfid\cable\DiT"
    # cut_bins(image_path, output_folder)

    real_image_path = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\test\good"
    generated_image_path = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\recon\train\good\recon"
    fid = cal_fid(real_image_path, generated_image_path)
    print(fid)

    # real = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\sfid\cable\real"
    # gen = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\sfid\cable\DiT"
    # sfid = cal_sfid(real, gen)
    # print(sfid)
