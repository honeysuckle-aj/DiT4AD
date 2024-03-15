import os
import torch
import random
import numpy as np
import time

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from einops import repeat, rearrange
from tqdm import tqdm

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF'
]

transform = transforms.Compose([transforms.Resize([256, 256]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                                                     inplace=True)])


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def make_bar(mask, bar_num=5):
    h, w = mask.shape
    for i in range(bar_num):
        bar_h = np.random.randint(h//10, h//2)
        bar_w = np.random.randint(w//10, w//2)
        bar_y = np.random.randint(0, h-bar_h)
        bar_x = np.random.randint(0, w-bar_w)
        mask[bar_y:bar_y+bar_h, bar_x:bar_x+bar_w] = np.ones((bar_h, bar_w))
    return mask


def make_mask(texture: np.ndarray, size: tuple):
    """
    make a mask using a texture image
    texture shape: (h w c)
    """
    mask = np.zeros(size)
    mask = make_bar(mask)
    texture_mask = texture * repeat(mask, "h w -> h w c", c=3)
    return texture_mask, mask


def add_mask(image, mask, texture_mask):
    image = np.array(image.resize(mask.shape))
    return Image.fromarray(np.uint8(image * (1 - repeat(mask, "h w -> h w c", c=3)) + texture_mask))


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    if isinstance(image_size, tuple):
        image_size = image_size[0]

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def random_rotate(image):
    if random.random() <= 0.25:
        return image
    deg = random.choice([Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270])
    return image.transpose(deg)


def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_textures(texture_path, image_size=(256, 256)):
    textures = []
    for filename in os.listdir(texture_path):
        if is_image_file(filename):
            texture_image = Image.open(os.path.join(texture_path, filename)).resize(image_size)
            textures.append(np.array(texture_image))
    assert len(textures) > 0
    return textures


def load_image_paths(image_path):
    images = [os.path.join(image_path, img) for img in os.listdir(image_path) if is_image_file(img)]
    return images


class MaskedDataset(Dataset):
    def __init__(self, image_path, textures, image_size=(256, 256), masked_ratio=0.6) -> None:
        self.image_path = load_image_paths(image_path)
        self.images = []
        self.masked_images = []
        self.textures = textures
        self.masks = []
        self.transform = transform

        for i in tqdm(range(len(self.image_path)), desc="loading images"):
            image_i = random_rotate(Image.open(self.image_path[i]))  # height * width * channel + rotation
            # image_i = rearrange(image_i, "h w c -> c h w") 
            self.images.append(self.transform(image_i))

            texture_mask, mask = make_mask(textures[i % len(textures)], size=image_size)
            self.masked_images.append(
                self.transform(add_mask(image_i, mask, texture_mask)))
            self.masks.append(mask)
        self.images = torch.tensor(np.array(self.images), dtype=torch.float)
        self.masked_images = torch.tensor(np.array(self.masked_images), dtype=torch.float)
        self.masks = torch.tensor(np.array(self.masks), dtype=torch.float)

    def __getitem__(self, index):
        return self.images[index], self.masked_images[index], self.masks[index]

    def __len__(self):
        return len(self.images)


class TestDataset(Dataset):
    def __init__(self, test_path, mask_gt=None, image_size=(256, 256)) -> None:
        self.good_image_path = load_image_paths(os.path.join(test_path, "good"))
        self.bad_image_path = load_image_paths(os.path.join(test_path, "bad"))
        self.x = []
        self.y = []
        self.transform = transform
        for i in tqdm(range(len(self.good_image_path)), desc="loading good images"):
            image_i = self.transform(random_rotate(Image.open(self.good_image_path[i])))
            # image_i = np.array(Image.open(self.good_image_path[i]).resize(image_size)) # height * width * channel
            # image_i = rearrange(image_i, "h w c -> c h w") 
            self.x.append(image_i)
            self.y.append(0)
        for i in tqdm(range(len(self.bad_image_path)), desc="loading bad images"):
            image_i = self.transform(random_rotate(Image.open(self.bad_image_path[i])))
            # image_i = np.array(Image.open(self.bad_image_path[i]).resize(image_size)) # height * width * channel
            # image_i = rearrange(image_i, "h w c -> c h w") 
            self.x.append(image_i)
            self.y.append(1)
        self.x = torch.tensor(np.array(self.x), dtype=torch.float)
        self.y = torch.tensor(np.array(self.y), dtype=torch.int64)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    textures = load_textures(r"dataset\textures")
    ad_dataset = MaskedDataset(image_path=r"dataset\capsule", textures=textures)
    dataloader = DataLoader(ad_dataset, batch_size=4, shuffle=False)
    p_bar = tqdm(enumerate(dataloader), desc="Training", unit="batch")
    for i, (img, m_img, m) in p_bar:
        time.sleep(0.2)
