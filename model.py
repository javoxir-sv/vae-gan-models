import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim:int=784, hidden_dim:int=400, latent_dim:int=2):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x:torch.Tensor):
        h1 = F.relu(self.fc1(x))
        mu = self.mu(h1)
        log_var = self.log_var(h1)
        return mu, log_var


    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # z = mu + std * epsilon
        if self.training:
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z:torch.Tensor) -> torch.Tensor:
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: torch.Tensor):
        x_flat = x.view(-1, self.input_dim)
        mu, log_var = self.encode(x_flat)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var



class VAESupervised(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 400, latent_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        
        # Encoder 
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim+10, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x:torch.Tensor):
        h1 = F.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        log_var = self.fc_logvar(h1)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        # z = mu + std * epsilon
        if self.training:
            std = torch.exp(0.5*log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z:torch.Tensor, digit_labels:torch.Tensor) -> torch.Tensor:
        z = torch.cat([z, digit_labels], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x: torch.Tensor, digit_labels:torch.Tensor):
        x_flat = x.view(-1, self.input_dim)
        mu, log_var = self.encode(x_flat)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, digit_labels)
        return recon_x, mu, log_var



class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.Linear(2, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 784),
                nn.Tanh()
                )

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.Linear(784, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256,  1),
                nn.Sigmoid()
                )

    def forward(self, x):
        return self.main(x)


class DiscriminatorNoSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
                nn.Linear(784, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256,  1),
                )

    def forward(self, x):
        return self.main(x)


class GeneratorSupervised(nn.Module):
    def __init__(self):
        super().__init__()


        self.noise_em = nn.Sequential(
                nn.Linear(2, 128),
                nn.LeakyReLU(0.2)
                )
        self.label_em = nn.Sequential(
                nn.Linear(10, 128),
                nn.LeakyReLU(0.2)
                )
        self.main = nn.Sequential(
                nn.Linear(256, 512),
                nn.LeakyReLU(0.2),
                nn.Linear(512, 784),
                nn.Tanh()
                )

    def forward(self, noise, digit_labels):
        noise_em = self.noise_em(noise)
        label_em = self.label_em(digit_labels)
        input = torch.cat([noise_em, label_em], dim=1)
        return self.main(input)


class DiscriminatorSupervised(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_em = nn.Sequential(
                nn.Linear(784, 256),
                nn.LeakyReLU(0.2)
                )
        self.label_em = nn.Sequential(
                nn.Linear(10, 256),
                nn.LeakyReLU(0.2)
                )
        self.main = nn.Sequential(
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2),
                nn.Linear(256,  1),
                nn.Sigmoid()
                )

    def forward(self, x, digit_labels):
        image_em = self.image_em(x)
        label_em = self.label_em(digit_labels)
        input = torch.cat([image_em, label_em], dim=-1)
        return self.main(input)
