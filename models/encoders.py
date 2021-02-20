import torch
from torch import nn

class VAE(nn.Module):
    def __init__(self, z_dims):
        super().__init__()
        self.z_dims = z_dims

        ## ENCODER ##
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.mu_fc = nn.Linear(6272, self.z_dims)
        self.var_fc = nn.Linear(6272, self.z_dims)

        ## DECODER ##
        self.decoder_input = nn.Linear(self.z_dims, 6272)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=3, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=7, stride=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=1, bias=False),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.mu_fc(x)
        var = self.var_fc(x)
        return mu, var

    def reparam(self, mu, var):
        std = torch.exp(var / 2)
        e = torch.randn_like(std)
        return e * std + mu

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, 128, 7, 7)
        x = self.decoder(x)
        return x

    def encode_state(self, x):
        mu, var = self.encode(x)
        z = self.reparam(mu, var)
        return z

    def forward(self, x):
        mu, var = self.encode(x)
        z = self.reparam(mu, var)
        x_hat = self.decode(z)
        return x_hat, mu, var

# test = VAE(64)
#
# n = torch.rand(size=(1, 1, 224, 224), dtype=torch.float)
#
# xhat, _, _ = test(n)
# print(xhat.shape)