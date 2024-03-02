from sklearn.decomposition import PCA

import numpy as np
from PIL import Image


def denoise(img):
    img = np.array(img)
    img_shape = img.shape
    if len(img_shape) == 2:
        h, w = img_shape
        channels = 1
        flattened_image = img.reshape((h * w, 1))
    else:
        h, w, channels = img.shape
        flattened_image = img.reshape((h * w, channels))
    pca = PCA(n_components=0.3)
    pca_result = pca.fit_transform(flattened_image)
    denoised_img = pca.inverse_transform(pca_result)
    if len(img_shape) == 2:
        denoised_img = denoised_img.reshape((h, w))
    else:
        denoised_img.reshape((h, w, channels))
    return Image.fromarray(denoised_img)


def add_latent_noise(img, mean=0, sigma=1):
    img = np.array(img)
    img_shape = img.shape
    if len(img_shape) == 2:
        h, w = img_shape
        channels = 1
        flattened_image = img.reshape((h * w, 1))
    else:
        h, w, channels = img.shape
        flattened_image = img.reshape((h * w, channels))
    pca = PCA(n_components=0.9)
    pca_result = pca.fit_transform(flattened_image)
    noise = np.random.normal(mean, sigma, pca_result.shape)
    noised_img = pca.inverse_transform(pca_result + noise)
    if len(img_shape) == 2:
        noised_img = noised_img.reshape((h, w))
    else:
        noised_img.reshape((h, w, channels))
    return Image.fromarray(noised_img)


def add_noise(img, mean=0, sigma=1):
    img = np.array(img)
    noise = np.random.normal(mean, sigma, img.shape)
    return Image.fromarray(img + noise)


def get_concat_h(im1, im2):
    dst = Image.new('L', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def partition_patches(image, patch_size, channel=3):
    height, width = image.size
    if height // patch_size != height / patch_size or width // patch_size != width / patch_size:
        print("size error")
        return
    patches = [image[i:i + patch_size, j:j + patch_size, :] for i in range(0, image.shape[0], patch_size)
               for j in range(0, image.shape[1], patch_size)]
    return np.array(patches)


def component_patches(patches, patch_size, height, width, channel=3):
    # image = np.zeros((height, width, channel))
    # for i in range(height//patch_size):
    #     for j in range(width//patch_size):
    #         image[i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size,:] = \
    #             patches[i*(width//patch_size)+j].reshape(patch_size,patch_size,channel)
    reconstructed_image = np.zeros((height, width, channel))
    for i, patch in enumerate(patches):
        row = (i // (image.shape[1] // patch_size)) * patch_size
        col = (i % (image.shape[1] // patch_size)) * patch_size
        reconstructed_image[row:row + patch_size, col:col + patch_size, :] = patch.reshape((patch_size, patch_size, 3))

    return reconstructed_image


if __name__ == '__main__':
    # iris = datasets.load_iris()['data']
    # 加载图像并转换为PyTorch张量
    image_path = r"E:\DataSets\AnomalyDetection\mvtec_anomaly_detection\capsule\train\good\000.png"

    # 读取RGB图像
    image = Image.open(image_path).convert('L')
    # height, width = image.size
    # patches = partition_patches(image, patch_size=10)
    # noised_patches = []
    # for patch in patches:
    #     noised_patches.append(add_latent_noise(patch, sigma=2))
    # noised_image = component_patches(noised_patches, 10, height, width)
    direct_noised_image = add_noise(image, mean=0, sigma=256)
    latent_noised_image = add_latent_noise(image, mean=0, sigma=256)
    noised_image = get_concat_h(direct_noised_image, latent_noised_image)
    # noised_image.show()
    denoised_image = denoise(latent_noised_image)
    clean_image = get_concat_h(image, denoised_image)
    # clean_image.show()
    result = get_concat_h(clean_image, noised_image)
    result.show()
    # 显示原始图像和降噪后的图像
    # residual = np.abs(noised_image - image)

    # output = np.concatenate((image, noised_image), axis=1)
    # print(noised_image.shape, image.shape, output.shape)
    # print(residual)
    # cv2.imwrite('output.png',output.astype(np.uint8))
    # cv2.imshow('or',output)
    # cv2.waitKey()
    # Image.fromarray(output.astype(np.uint8), mode='RGB').show()
