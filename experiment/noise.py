from sklearn.decomposition import PCA

import numpy as np
from PIL import Image

def add_latent_noise(patch,mean=0, sigma=1):
    height, width, channels = patch.shape
    flattened_image = patch.reshape((height*width,channels))
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(flattened_image)
    noise = np.random.normal(mean,sigma, pca_result.shape)
    noised_patch = pca.inverse_transform(pca_result+noise)
    return noised_patch

def partition_patches(image, patch_size, channel=3):
    height, width, channels = image.shape
    if height//patch_size != height/patch_size or width//patch_size != width/patch_size:
        print("size error")
        return
    patches = [image[i:i+patch_size, j:j+patch_size, :] for i in range(0, image.shape[0], patch_size) 
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
        reconstructed_image[row:row+patch_size, col:col+patch_size, :] = patch.reshape((patch_size, patch_size, 3))

    return reconstructed_image


if __name__ == '__main__':

    # iris = datasets.load_iris()['data']
    # 加载图像并转换为PyTorch张量
    image_path = "dataset/xin.png"

    # 读取RGB图像
    image = cv2.imread(image_path)
    image = cv2.resize(image, (100,100))
    height, width, channel = image.shape
    patches = partition_patches(image, patch_size=10)
    noised_patches = []
    for patch in patches:
        noised_patches.append(add_latent_noise(patch,sigma=2))
    noised_image = component_patches(noised_patches, 10, height, width)

    # 显示原始图像和降噪后的图像
    residual = np.abs(noised_image - image)
    
    output = np.concatenate((image, noised_image, residual),axis=1)
    # print(noised_image.shape, image.shape, output.shape)
    print(residual)
    # cv2.imwrite('output.png',output.astype(np.uint8))
    # cv2.imshow('or',output)
    # cv2.waitKey()
    Image.fromarray(output.astype(np.uint8),mode='RGB').show()





