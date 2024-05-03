import torch
import torch.nn.functional as fn
import matplotlib.pyplot as plt


def draw_fn():
    x = torch.linspace(-3, 3, 1000)
    y_relu = fn.relu(x)
    y_elu = fn.elu(x)
    y_silu = fn.silu(x)
    y_gelu = fn.gelu(x)

    plt.plot(x, y_relu, c='blue', label='ReLU')
    plt.plot(x, y_elu, c='orange', label='ELU')
    plt.plot(x, y_silu, c='purple', label='SiLU')
    plt.plot(x, y_gelu, c='red', label='GELU')
    plt.legend()
    plt.savefig("act_fn.png")


if __name__ == "__main__":
    draw_fn()
