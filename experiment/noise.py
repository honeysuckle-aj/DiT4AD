import random

from sklearn.decomposition import PCA

import numpy as np
from einops import rearrange
from PIL import Image, ImageDraw


def add_latent_noise(patch, mean=0, sigma=1):
    height, width, channels = patch.shape
    flattened_image = patch.reshape((height * width, channels))
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(flattened_image)
    noise = np.random.normal(mean, sigma, pca_result.shape)
    noised_patch = pca.inverse_transform(pca_result + noise)
    return noised_patch


def split_image_into_patches(image_path, patch_size):
    # 打开图片
    image = Image.open(image_path)

    # 获取图片的宽度和高度
    width, height = image.size

    # 计算水平和垂直方向上的patch数量
    num_patches_horizontal = width // patch_size
    num_patches_vertical = height // patch_size

    # 创建一个空列表来存储所有的patch
    patches = []

    # 循环遍历图像并分割成patch
    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            # 计算当前patch的左上角和右下角坐标
            left = j * patch_size
            upper = i * patch_size
            right = left + patch_size
            lower = upper + patch_size
            # 从原始图像中提取当前patch
            patch = image.crop((left, upper, right, lower))
            # 将当前patch添加到列表中
            patches.append(patch)

    return patches


def combine_patches_into_image(patch_list, original_image_size, patch_size):
    # 获取原始图像的宽度和高度
    original_width, original_height = original_image_size

    # 创建一个空白的图像对象，大小与原始图像相同
    reconstructed_image = Image.new("RGB", (original_width, original_height))

    # 计算水平和垂直方向上的patch数量
    num_patches_horizontal = original_width // patch_size
    num_patches_vertical = original_height // patch_size

    # 循环遍历patch列表，并将每个patch粘贴回原始图像
    for i in range(num_patches_vertical):
        for j in range(num_patches_horizontal):
            # 计算当前patch在原始图像中的位置
            left = j * patch_size
            upper = i * patch_size
            right = left + patch_size
            lower = upper + patch_size
            # 获取当前patch
            patch = patch_list[(i * num_patches_horizontal + j + random.randint(1, 10)) % len(patch_list)]
            # 将当前patch粘贴回原始图像
            reconstructed_image.paste(patch, (left, upper, right, lower))

    return reconstructed_image


if __name__ == '__main__':
    # img = np.zeros((256, 256))
    # img = Image.fromarray(img.astype(np.int64), "RGB")
    # draw = ImageDraw.Draw(img)
    # x = getattr(draw, 'ellipse')
    # x(xy=[0, 0, 60, 50], fill="white")
    #
    # img.save("../dataset/textures/08.png", format="png")

    image = Image.open(r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\test\good\007.png")
    h, w = image.size
    patches = split_image_into_patches(r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\test\good\007.png",
                                       128)
    re_image = combine_patches_into_image(patches, (h, w), 128)
    re_image.save("../dataset/textures/08.png")
