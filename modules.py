import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_latent_space(model, dataloader):
    model.eval()
    all_mu, all_labels = [], []
    with torch.no_grad():
        for data, label in dataloader:
            mu, _ = model.encode(data.to(device).view(-1, model.input_dim))
            all_mu.append(mu.cpu().numpy())
            all_labels.append(label.numpy())

    all_mu = np.concatenate(all_mu, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(all_mu[:, 0], all_mu[:, 1], c=all_labels, cmap='tab10', s=2)
    plt.colorbar(scatter)
    plt.title("2D laten space of MNIST")
    plt.tight_layout()
    plt.show()


def plot_manifold(model):
    n = 15
    digit_size = 28
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)
    figure = np.zeros((digit_size * n, digit_size * n))

    model.eval()
    with torch.no_grad():
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
                x_decoded = model.decode(z_sample)
                digit = x_decoded.reshape(digit_size, digit_size).cpu().numpy()
                figure[i*digit_size: (i+1)*digit_size,
                       j*digit_size: (j+1)*digit_size] = digit

    plt.figure(figsize=(8, 8))
    plt.imshow(figure, cmap="gray")
    plt.tight_layout()
    plt.show()


def plot_supervised_manifold(model):
    model.eval()
    n_styles = 15
    n_digits = 10
    digit_size = 28
    
    z_values = torch.linspace(-4, 4, n_styles).view(-1, 1).to(device)
    figure = np.zeros((digit_size * n_digits, digit_size * n_styles))
    
    with torch.no_grad():
        for digit_idx in range(n_digits):
            label_single = F.one_hot(torch.tensor([digit_idx]), num_classes=10).float().to(device)
            label_batch = label_single.repeat(n_styles, 1)
            decoded = model.decode(z_values, label_batch)
            
            for style_idx in range(n_styles):
                digit_img = decoded[style_idx].reshape(digit_size, digit_size).cpu().numpy()
                row_start = digit_idx * digit_size
                row_end = (digit_idx + 1) * digit_size
                col_start = style_idx * digit_size
                col_end = (style_idx + 1) * digit_size
                figure[row_start:row_end, col_start:col_end] = digit_img

    plt.figure(figsize=(10, 7))
    plt.imshow(figure, cmap='gray')
    plt.title("Supervised VAE: 10 digits and 15 latent values")
    plt.xlabel("Latent variable z (from -4 to +4)")
    plt.yticks(
        ticks=np.arange(14, 10 * 28, 28), 
        labels=[f"digit {i}" for i in range(10)]
    )
    plt.xticks([])
    plt.tight_layout()
    plt.show()



def plot_gan_manifold(model, limit=3):
    n = 15
    digit_size = 28
    grid_x = np.linspace(-limit, limit, n)
    grid_y = np.linspace(-limit, limit, n)
    figure = np.zeros((digit_size * n, digit_size * n))

    model.eval()
    with torch.no_grad():
        for i, xi in enumerate(grid_x):
            for j, yi in enumerate(grid_y):
                z_sample = torch.tensor([[xi, yi]], dtype=torch.float).to(device)
                x_decoded = model(z_sample)
                digit = x_decoded.reshape(digit_size, digit_size).cpu().numpy()
                figure[i*digit_size: (i+1)*digit_size,
                       j*digit_size: (j+1)*digit_size] = digit

    plt.figure(figsize=(8, 8))
    plt.imshow(figure, cmap="gray")
    plt.tight_layout()
    plt.show()



class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

# Helper function
def grad_reverse(x, alpha=1.0):
    return GradientReversalFunction.apply(x, alpha)


def plot_supervised_manifold_gan(model):
    model.eval()
    n_styles = 15
    n_digits = 10
    digit_size = 28
    
    noise = torch.randn(n_styles, 2).to(device)
    figure = np.zeros((digit_size * n_digits, digit_size * n_styles))
    
    with torch.no_grad():
        for digit_idx in range(n_digits):
            label_single = F.one_hot(torch.tensor([digit_idx]), num_classes=10).float().to(device)
            label_batch = label_single.repeat(n_styles, 1)
            decoded = model(noise, label_batch)
            for style_idx in range(n_styles):
                digit_img = decoded[style_idx].reshape(digit_size, digit_size).cpu().numpy()
                row_start = digit_idx * digit_size
                row_end = (digit_idx + 1) * digit_size
                col_start = style_idx * digit_size
                col_end = (style_idx + 1) * digit_size
                figure[row_start:row_end, col_start:col_end] = digit_img

    plt.figure(figsize=(10, 7))
    plt.imshow(figure, cmap='gray')
    plt.title("Supervised GAN: 10 digits and 15 latent values")
    plt.yticks(
        ticks=np.arange(14, 10 * 28, 28), 
        labels=[f"digit {i}" for i in range(10)]
    )
    plt.xticks([])
    plt.tight_layout()
    plt.show()
