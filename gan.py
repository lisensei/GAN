import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import PILToTensor
from PIL import Image
import torchvision.transforms.functional as T


class Generator(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=8, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=13, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=13, out_channels=21, kernel_size=3)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=21, out_channels=34, kernel_size=3)
        self.fc = nn.Linear(in_features=12096, out_features=55)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = x.reshape(x.size()[0], -1)
        out = self.fc(x)
        return out


dataset = CIFAR10(root="./data", download=True, transform=PILToTensor())
dataloader = DataLoader(dataset, 1)

generator = Generator()
x = torch.randn(size=(1, 3, 32, 32))
y = generator(x)
