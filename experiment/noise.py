import os
import random
import shutil

from sklearn.decomposition import PCA

import numpy as np
from einops import rearrange, repeat
from PIL import Image, ImageDraw
import opensimplex as simplex
from tqdm import tqdm


def add_simplex_noise(img, texture):
    h, w = img.size
    pass


def generate_2D_img(im, texture, width, height, feature_size, threshold=-0.5, filename='mask_image_perlin.png'):
    print('Generating 2D image...')
    if im is None:
        im = Image.new('L', (width, height))
    if texture is None:
        texture = Image.new('L', (width, height), 'white')
    for y in tqdm(range(0, height)):
        for x in range(0, width):
            value = simplex.noise2(x / feature_size, y / feature_size)
            if value < threshold:
                im.putpixel((x, y), value=texture.getpixel((y, x)))
    im.save(filename)


def generate_random_noise(im, texture, width, height, threshold=0.5):
    for y in range(0, height):
        for x in range(0, width):
            value = random.random()
            if value < threshold:
                im.putpixel((x, y), value=texture.getpixel((y, x)))
        im.save(f'mask_image_random.png')


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


def make_masked_image(mask, texture, image):
    mask = np.array(mask) // 255
    texture = np.array(texture)
    image = np.array(image)
    image_base = image * (1 - mask)
    image_t = texture * mask
    return Image.fromarray(image_base + image_t)


def add_noise(image, ratio):
    image = np.array(image)
    h, w, c = image.shape
    noise = np.random.normal(scale=255, size=(h, w, c))
    noised = image * ratio + noise * (1 - ratio)
    return Image.fromarray(noised.astype(np.uint8))


def move_mask():
    folder = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable"
    mask_folder = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\recon\train\good\mask"
    image_folder = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\recon\train\good\origin"
    idx = 237
    for fd in tqdm(os.listdir(os.path.join(folder, "ground_truth"))):
        if fd in ["good", "bad"]:
            continue
        for f in os.listdir(os.path.join(folder, "ground_truth", fd)):
            mask_fn = os.path.join(folder, "ground_truth", fd, f)
            image_fn = os.path.join(folder, "test", fd, f[:3] + ".png")
            new_mask = os.path.join(mask_folder, f"train_{idx}.png")
            new_image = os.path.join(image_folder, f"train_{idx}.png")
            mask = Image.open(mask_fn).resize((256, 256))
            mask = np.array(mask)
            # img[np.where(img > 0)] = 1
            mask = mask[:, :, np.newaxis]
            mask2 = 255 - mask
            mask = np.concatenate((mask, mask2), axis=2)
            mask = Image.fromarray(mask)
            mask.save(new_mask)
            shutil.copy(image_fn, new_image)
            idx += 1

def resize_images(folder1, folder2, size=256):
    for f in tqdm(os.listdir(folder1)):
        fn = os.path.join(folder1, f)
        im = Image.open(fn).resize((size, size))
        im.save(os.path.join(folder2, f))
def move_data(folder):
    for fd in tqdm(os.listdir(folder)):
        if fd not in ['bad', 'good']:
            for f in os.listdir(os.path.join(folder, fd)):
                fn = os.path.join(folder, fd, f)
                new_fn = os.path.join(folder, "bad", f"{fd}_{f}")
                shutil.copy(fn, new_fn)


if __name__ == '__main__':
    # pass
    move_data(r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\ground_truth")
    # move_mask()
    # resize_images(r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\recon\train\good\origin2",
    #               r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\recon\train\good\origin")
    # img = np.zeros((256, 256))
    # img = Image.fromarray(img.astype(np.int64), "RGB")
    # draw = ImageDraw.Draw(img)
    # x = getattr(draw, 'ellipse')
    # y = getattr(draw, 'rectangle')
    # y(xy=[60, 50, 80, 80], fill="white")
    # x(xy=[110, 120, 150, 140], fill="white")
    # img.save("mask_pred.png", format="png")
    # texture = Image.open(r"D:\Projects\DiT4AD\dataset\textures\04.jpg").resize((256,256))
    # mask = Image.open("mask.png").resize((256,256))
    # image = Image.open(r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\test\good\001.png").resize((256,256))
    # masked = make_masked_image(mask, texture, image)
    # masked.save("masked_img_geo.png", format='png')
    # generate_2D_img(None, None, 1024, 1024, feature_size=32, filename="big_perlin.png")
    # generate_2D_img(image, texture, image.size[0], image.size[1], feature_size=32, filename="big_perlin.png")
    # generate_random_noise(image, texture, image.size[0], image.size[1], threshold=0.2)
    # masked = Image.open("masked_img.png")
    # for i in range(5):
    #
    #     noised = add_noise(image, 1-i/15)
    #     noised.save(f"noised{i}.png", format='png')
    # generate_2D_img(256, 256, 64)
    # image = Image.open(r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\screw\test\good\007.png")
    # h, w = image.size
    # patches = split_image_into_patches(r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\screw\test\good\007.png",
    #                                    128)
    # re_image = combine_patches_into_image(patches, (h, w), 128)
    # re_image.save("../dataset/textures/09.png")

    # img_path = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\cable\recon\train\good\mask\train_221.png"
    # img = Image.open(img_path)
    # print(img.size)
    # img_a = np.array(img)
    # print(img_a.shape)
