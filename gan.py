import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import PILToTensor
from PIL import Image
import torch.nn.functional as TF
import torchvision.transforms.functional as T
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-image_height", type=int, default=32)
parser.add_argument("-image_width", type=int, default=32)
parser.add_argument("-latent_dim", type=int, default=32)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=16)
parser.add_argument("-learning_rate", type=float, default=5e-4)
args = parser.parse_args()


class Generator(nn.Module):
    def __init__(self, latent_dim, out_features=64) -> None:
        super().__init__()
        self.first_layer_out_width = int(out_features**0.5)
        self.oneD2twoD = nn.Linear(latent_dim, out_features)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=1, out_channels=3, kernel_size=3)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=3, out_channels=3, kernel_size=3)
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=3, out_channels=3, kernel_size=3)
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=3, out_channels=3, kernel_size=3)
        self.deconv5 = nn.ConvTranspose2d(
            in_channels=3, out_channels=3, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.oneD2twoD(
            x).reshape(-1, 1, self.first_layer_out_width, self.first_layer_out_width)
        x = TF.relu(self.deconv1(x))
        x = TF.relu(self.deconv2(x))
        x = TF.relu(self.deconv3(x))
        x = TF.relu(self.deconv4(x))
        x = TF.relu(self.deconv5(x))
        return TF.sigmoid(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=5, kernel_size=3)
        self.conv2 = nn.Conv2d(
            in_channels=5, out_channels=13, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=13, out_channels=34, kernel_size=3)
        self.conv4 = nn.Conv2d(
            in_channels=34, out_channels=55, kernel_size=3, stride=2)
        self.fc = nn.Linear(in_features=55*25, out_features=2)

    def forward(self, x):
        x = TF.relu(self.conv1(x))
        x = TF.relu(self.conv2(x))
        x = TF.relu(self.conv3(x))
        x = TF.relu(self.conv4(x))
        x = x.flatten(start_dim=1)
        return self.fc(x)


dataset = CIFAR10(root="./data", download=False, transform=PILToTensor())
dataloader = DataLoader(dataset, args.batch_size)
generator = Generator(args.latent_dim)
discriminator = Discriminator()

generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=args.learning_rate)
discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=args.learning_rate)

generator_loss_function = nn.CrossEntropyLoss()
discriminator_loss_function = nn.CrossEntropyLoss()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
generator.to(DEVICE)

num_gen_parameters = sum([param.numel() for param in generator.parameters()])
num_dis_parameters = sum([param.numel()
                         for param in discriminator.parameters()])

for k, v in args.__dict__.items():
    print(f"{k} : {v}")

print(f"number of generator parameters: {num_gen_parameters}")
print(f"number of discriminator parameters: {num_dis_parameters}")
for e in range(args.epochs):
    # train generator
    for i, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        discriminator.eval()
        gen_latents = torch.randn(size=(args.batch_size, args.latent_dim))
        fake_images = generator(gen_latents)
        fake_labels = torch.ones(
            size=(args.batch_size,), dtype=torch.int64, device=DEVICE)
        generator_score = discriminator(fake_images)
        generator_loss = generator_loss_function(generator_score, fake_labels)
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

        # train discriminator
        generator.eval()
        dis_latents = torch.randn(size=(args.batch_size, args.latent_dim))
        fake_images = generator(dis_latents)
        fake_labels = torch.zeros(
            size=(args.batch_size,), dtype=torch.int64, device=DEVICE)
        real_labels = torch.ones(
            size=(args.batch_size,), dtype=torch.int64, device=DEVICE)
        discriminator_input = torch.cat([fake_images, images], dim=0)
        all_labels = torch.cat([fake_labels, real_labels], dim=0)
        discriminator_score = discriminator(discriminator_input)
        discriminator_loss = discriminator_loss_function(
            discriminator_score, all_labels)
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()
