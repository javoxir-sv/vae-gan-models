import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from modules import grad_reverse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def vae_loss_function(recon_x:torch.Tensor, x:torch.Tensor, mu:torch.Tensor, log_var:torch.Tensor, input_dim:int=784):
    recon_loss = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss


def train_gan(generator, discriminator, dataloader, criterion, optG, optD, num_epochs):
    lossesD, lossesG = [], []
    for _ in range(num_epochs):
        avg_lossG, avg_lossD = 0, 0
        for real_images, _ in dataloader:
            real_images = real_images.to(device).view(-1, 784)
            real_images = (real_images - 0.5) / 0.5
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1).to(device) * 0.9
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # Training the Discriminator
            optD.zero_grad()
            output_real = discriminator(real_images)
            lossD_real = criterion(output_real, real_labels)

            noise = torch.randn(batch_size, 2).to(device)
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            lossD_fake = criterion(output_fake, fake_labels)
            lossD = lossD_real + lossD_fake 
            avg_lossD += lossD.item()
            lossD.backward()
            optD.step()

            # Training the Generator
            optG.zero_grad()
            output_fake_G = discriminator(fake_images)
            lossG = criterion(output_fake_G, real_labels)
            avg_lossG += lossG.item()
            lossG.backward()
            optG.step()

        lossesD.append(avg_lossD / len(dataloader))
        lossesG.append(avg_lossG / len(dataloader))
    return lossesD, lossesG


def train_gan_grl(generator, discriminator, dataloader, criterion, optG, optD, num_epochs, alpha=0.1, gclip=False):
    lossesD = []
    for epoch in range(num_epochs):
        avg_lossD = 0
        for real_images, _ in dataloader:
            real_images = real_images.to(device).view(-1, 784)
            real_images = (real_images - 0.5) / 0.5
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1).to(device) * 0.9
            fake_labels = torch.zeros(batch_size, 1).to(device)

            optD.zero_grad()
            optG.zero_grad()

            # This one trains both Generator and Discriminator at the same time.
            # Because their gradients are connected through GradientReverseLayer.
            output_real = discriminator(real_images)
            lossD_real = criterion(output_real, real_labels)
            noise = torch.randn(batch_size, 2).to(device)

            fake_images = generator(noise)
            fake_images_reversed = grad_reverse(fake_images, alpha=alpha)
            output_fake = discriminator(fake_images_reversed)

            lossD_fake = criterion(output_fake, fake_labels)
            lossD = lossD_real + lossD_fake 
            lossD.backward()

            if gclip:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)

            optD.step()
            optG.step()

            avg_lossD += lossD.item()

        lossesD.append(avg_lossD / len(dataloader))
        print(f"Epoch [{epoch+1:02d}/{num_epochs}] | Loss: {lossesD[-1]:.4f}")
    return lossesD



def train_gan_supervised(generator, discriminator, dataloader, criterion, optG, optD, num_epochs):
    lossesD, lossesG = [], []
    for _ in range(num_epochs):
        avg_lossG, avg_lossD = 0, 0
        for real_images, labels in dataloader:
            real_images = real_images.to(device).view(-1, 784)
            real_images = (real_images - 0.5) / 0.5
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1).to(device) * 0.9
            fake_labels = torch.zeros(batch_size, 1).to(device)

            digit_labels = F.one_hot(labels, num_classes=10).float().to(device)
            noise = torch.randn(batch_size, 2).to(device)

            # Training the Discriminator
            optD.zero_grad()
            output_real = discriminator(real_images, digit_labels)
            lossD_real = criterion(output_real, real_labels)

            fake_images = generator(noise, digit_labels)
            output_fake = discriminator(fake_images.detach(), digit_labels)
            lossD_fake = criterion(output_fake, fake_labels)

            wrong_labels = digit_labels[torch.randperm(batch_size)]
            output_wrong = discriminator(real_images, wrong_labels)
            lossD_wrong = criterion(output_wrong, fake_labels)

            lossD = lossD_real + lossD_fake + lossD_wrong 
            avg_lossD += lossD.item()
            lossD.backward()
            optD.step()

            # Training the Generator
            optG.zero_grad()
            output_fake_G = discriminator(fake_images, digit_labels)
            lossG = criterion(output_fake_G, real_labels)
            avg_lossG += lossG.item()
            lossG.backward()
            optG.step()

        lossesD.append(avg_lossD / len(dataloader))
        lossesG.append(avg_lossG / len(dataloader))
    return lossesD, lossesG

