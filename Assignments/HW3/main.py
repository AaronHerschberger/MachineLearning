import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader  

import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets  
from tqdm import tqdm

inputFile = "data/even_mnist.csv/"

# dataset = torch.load(inputFile)
# print(dataset)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 20*2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return eps.mul(std).add_(mu)
    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, 784))
        mu = mu_logvar[:, :20]
        logvar = mu_logvar[:, 20:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

def prepare_data(rawcsv):
    return rawcsv

def main():
    print("Hello World!")

main()