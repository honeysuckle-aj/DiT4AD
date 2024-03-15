import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torch import nn


class SegCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_down1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.conv_down2 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv_up = nn.ConvTranspose2d(in_channels=16, out_channels=6, kernel_size=2, stride=2)
        self.conv_out = nn.Conv2d(in_channels=6, out_channels=out_channels, kernel_size=3, padding=1)

        self.nn_seq = nn.Sequential(self.conv_down1, self.conv_down2, self.pool1, self.conv_up, self.conv_out,
                                    nn.Sigmoid())

    def forward(self, x):
        return self.nn_seq(x)


if __name__ == '__main__':
    seg = SegCNN(in_channels=9, out_channels=1)
    image = np.array(Image.open("../dataset/textures/01.jpg").resize((256, 256)))
    input_image = np.concatenate((image, image, image), axis=2)
    input_image = torch.tensor(input_image, dtype=torch.float)
    input_image = rearrange(input_image, "h w c -> c h w")
    with torch.no_grad():
        output_image = seg(input_image)
    Image.fromarray(np.array(output_image.squeeze(0), dtype=np.uint8)).show()
    print('finish')
