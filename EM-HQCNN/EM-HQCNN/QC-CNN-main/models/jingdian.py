import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

n_class = 3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 4, 2, stride=2)  # 14x14 -> 7x7
        self.ln1 = nn.LayerNorm([4, 7, 7], elementwise_affine=True)
        self.act1 = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(4*7*7, 12)
        self.ln2 = nn.LayerNorm(12)
        self.act2 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(12, n_class)

    def forward(self, x):
        x = x.view(x.shape[0], 1, 14, 14)
        x = self.conv(x)
        x = self.ln1(x)
        x = self.act1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.ln2(x)
        x = self.act2(x)
        x = self.fc2(x)
        return x
