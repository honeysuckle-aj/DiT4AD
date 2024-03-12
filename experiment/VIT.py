# from https://github.com/lucidrains/vit-pytorch
import numpy as np
import torch
from timm.models.vision_transformer import PatchEmbed, Block
from torch import nn
from PIL import Image
from einops import rearrange, repeat

from ad_dataset import pair


def train_seg_loss(model, criterion, latent_x, pred_latent_x, y, vae, ratio=0.8):
    with torch.no_grad():
        inter_x = vae.decode((ratio * pred_latent_x + (1 - ratio) * latent_x) / 0.18215).sample
        x = vae.decode(latent_x / 0.18215).sample
        pred_x = vae.decode(pred_latent_x / 0.18215).sample
    seg_input = torch.cat((x, pred_x, inter_x), dim=1)
    pred_y = model(seg_input)
    loss = criterion(pred_y, y)
    return pred_y, loss


if __name__ == '__main__':
    v = ViT(
        input_size=256,
        patch_size=16,
        output_size=256,
        hidden_size=1024,
        depth=9,
        num_heads=16,
        dropout=0.1,
        emb_dropout=0.1,
        in_channels=9
    )
    image = np.array(Image.open("../dataset/textures/01.jpg").resize((256, 256)))
    input_image = np.concatenate((image, image, image), axis=2)
    input_image = torch.tensor(input_image, dtype=torch.float)
    input_image = rearrange(input_image, "h w c -> 1 c h w")
    with torch.no_grad():
        output_image = v(input_image)
    Image.fromarray(np.array(output_image.squeeze(0), dtype=np.uint8)).show()
    print('a')
