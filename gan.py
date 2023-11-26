import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import PILToTensor
from PIL import Image
import torch.nn.functional as TF
import torchvision.transforms as T
import torchvision.transforms.functional as F
import sys
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
parser = ArgumentParser()
parser.add_argument("-image_height", type=int, default=32)
parser.add_argument("-image_width", type=int, default=32)
parser.add_argument("-latent_dim", type=int, default=32)
parser.add_argument("-epochs", type=int, default=100)
parser.add_argument("-batch_size", type=int, default=128)
parser.add_argument("-learning_rate", type=float, default=5e-4)
parser.add_argument("-generator_training_frequency", type=int, default=10)
parser.add_argument("-logdir", type=str, default="runs")
args = parser.parse_args()

if sys.platform[:3] == "win":
    run_time = datetime.now().isoformat(timespec="seconds").replace(":", "-")
else:
    run_time = datetime.now().isoformat(timespec="seconds")
writer = SummaryWriter(log_dir=f"{args.logdir}/{run_time}")


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
        x = TF.sigmoid(self.deconv5(x))
        return x


class LinearGenerator(nn.Module):
    def __init__(self, latent_dim, image_height=32, image_width=32) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.l1 = nn.Linear(latent_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.l2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.l3 = nn.Linear(256, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.l4 = nn.Linear(512, 3*image_height*image_width)
        self.bn4 = nn.BatchNorm1d(3*image_height*image_width)

    def forward(self, x):
        x = TF.relu(self.bn1(self.l1(x)))
        x = TF.relu(self.bn2(self.l2(x)))
        x = TF.relu(self.bn3(self.l3(x)))
        x = TF.sigmoid(self.bn4(self.l4(x)))
        x = x.reshape(-1, 3, self.image_height, self.image_width)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, stride=2)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc = nn.Linear(in_features=256*25, out_features=2)

    def forward(self, x):
        x = TF.relu(self.bn1(self.conv1(x)))
        x = TF.relu(self.bn2(self.conv2(x)))
        x = TF.relu(self.bn3(self.conv3(x)))
        x = TF.relu(self.bn4(self.conv4(x)))
        x = x.flatten(start_dim=1)
        return self.fc(x)


dataset = CIFAR10(root="./data", download=False,
                  transform=T.Compose([PILToTensor(), T.ConvertImageDtype(torch.float32)]))
dataloader = DataLoader(dataset, args.batch_size, drop_last=True)
generator = LinearGenerator(args.latent_dim)
discriminator = Discriminator()

generator_optimizer = torch.optim.Adam(
    generator.parameters(), lr=args.learning_rate)
discriminator_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=args.learning_rate)

generator_loss_function = nn.CrossEntropyLoss()
discriminator_loss_function = nn.CrossEntropyLoss()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
generator.to(DEVICE)
discriminator.to(DEVICE)
num_gen_parameters = sum([param.numel() for param in generator.parameters()])
num_dis_parameters = sum([param.numel()
                         for param in discriminator.parameters()])

for k, v in args.__dict__.items():
    print(f"{k} : {v}")

print(f"number of generator parameters: {num_gen_parameters}")
print(f"number of discriminator parameters: {num_dis_parameters}")
plt.figure(1)
for e in range(args.epochs):
    # train generator
    for i, (images, _) in enumerate(dataloader):
        images = images.to(DEVICE)
        discriminator.eval()
        generator.train()

        for p in generator.parameters():
            p.requires_grad = True
        for p in discriminator.parameters():
            p.requires_grad = False
        for f in range(args.generator_training_frequency):
            gen_latents = torch.randn(
                size=(args.batch_size, args.latent_dim), device=DEVICE)
            fake_images = generator(gen_latents)
            fake_labels = torch.ones(
                size=(args.batch_size,), dtype=torch.int64, device=DEVICE)
            for p in discriminator.parameters():
                p.requires_grad = False
            generator_score = discriminator(fake_images)
            generator_loss = generator_loss_function(
                generator_score, fake_labels)
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        # train discriminator
        generator.eval()
        discriminator.train()
        for p in generator.parameters():
            p.requires_grad = False
        for p in discriminator.parameters():
            p.requires_grad = True
        dis_latents = torch.randn(
            size=(args.batch_size, args.latent_dim), device=DEVICE)
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
    for j, image in enumerate(fake_images):
        if j > 15:
            break
        image = F.to_pil_image(image)
        plt.subplot(4, 4, j+1)
        plt.imshow(image)
        plt.axis("off")
    writer.add_figure("generations", plt.figure(1), e)
    plt.clf()
    print(f"epoch: {e}")
