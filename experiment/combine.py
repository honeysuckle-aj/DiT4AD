import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


class Simple(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
        self.fn = nn.Sigmoid()

    def forward(self, x):
        return self.fn(self.linear(x))


class SimpleData(Dataset):
    def __init__(self, data):
        self.x = torch.tensor(data[:-1].T, dtype=torch.float)
        self.y = torch.tensor(data[-1].T, dtype=torch.float)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return len(self.y)


def train():
    data = np.loadtxt(r"D:\Projects\DiT4AD\samples\eval\output.txt")
    x = data[:-1].T
    y = data[-1].T
    model = Perceptron(max_iter=100000, n_iter_no_change=100, penalty='elasticnet')
    model.fit(x, y)
    # preds = model.predict(x)
    print(model.score(x, y))
    print(model.coef_)


if __name__ == '__main__':
    train()
