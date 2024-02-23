import os
import torch
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

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def make_mask(texture:np.ndarray, size:tuple, ratio=0.3):
    """
    make a mask using a texture image
    """
    mask = np.random.choice([0,1],size=size, p=[1-ratio, ratio])
    texture_mask = texture * repeat(mask, "h w -> c h w", c=3)
    return texture_mask, mask


def is_image_file(filename:str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def load_textures(texture_path, image_size=(256,256)):
    textures = []
    for filename in os.listdir(texture_path):
        if is_image_file(filename):
            texture = np.array(Image.open(os.path.join(texture_path, filename)).resize(image_size))
            textures.append(rearrange(texture, "h w c -> c h w"))
    assert len(textures) > 0
    return textures
    

def load_image_paths(image_path):
    images = [os.path.join(image_path,img) for img in os.listdir(image_path) if is_image_file(img)]
    return images

class MaskedDataset(Dataset):
    def __init__(self, image_path, textures, image_size=(256,256), masked_ratio=0.6) -> None:
        self.image_path = load_image_paths(image_path)
        self.images = []
        self.masked_images = []
        self.textures = textures
        self.masks = []
        for i in tqdm(range(len(self.image_path)), desc="loading images"):
            image_i = np.array(Image.open(self.image_path[i]).resize(image_size)) # height * width * channel
            image_i = rearrange(image_i, "h w c -> c h w") 
            self.images.append(image_i)
            if i % 10 < masked_ratio * 10:
                texture_mask, mask = make_mask(textures[i % len(textures)], size=image_size)           
                self.masked_images.append(image_i * (1 - repeat(mask, "h w -> c h w", c=3)) + texture_mask)
            else:
                mask = np.zeros(image_size)
                self.masked_images.append(image_i)          
            self.masks.append(mask)
        self.images = torch.tensor(np.array(self.images), dtype=torch.float)
        self.masked_images = torch.tensor(np.array(self.masked_images), dtype=torch.float)
        self.masks = torch.tensor(np.array(self.masks), dtype=torch.float)

    def __getitem__(self, index):
        return self.images[index], self.masked_images[index], self.masks[index]
    
    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    textures = load_textures(r"dataset\textures")
    ad_dataset = MaskedDataset(image_path=r"dataset\capsule", textures=textures)
    dataloader = DataLoader(ad_dataset, batch_size=4, shuffle=False)
    p_bar = tqdm(enumerate(dataloader), desc="Training", unit="batch")
    for i,(img, m_img, m) in p_bar:
        time.sleep(0.2)


        


