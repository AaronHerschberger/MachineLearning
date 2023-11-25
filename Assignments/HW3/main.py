import torch
import torch.nn as nn
import torch.nn.functional as funcs
from torch.utils.data import DataLoader  

import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder

from tqdm import tqdm

from VAEclass import myVAEdef

inputFile = "./data/even_mnist.csv/"

myTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.(root=inputFile, transform=myTransform)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        
        # Encode function
        self.in2hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 20*2)
        )
        
        # Decode function
        self.latent2hidden = nn.Sequential(
            nn.Linear(20, 400),
            nn.ReLU(),
            nn.Linear(400, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid()
        )
        
    def endode(self, x):
        hidden = self.in2hidden(x)
        mu = funcs.relu(hidden[:, :20])
        
        return mu
    
    def decode(self, z):
        hidden = self.latent2hidden(z)
        output = funcs.sigmoid(hidden)
        
        return output
    
    def forward(self, x):
        mu_logvar = self.encoder(x.view(-1, 784))
        mu = mu_logvar[:, :20]
        logvar = mu_logvar[:, 20:]
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.rand_like(std)
    #     return eps.mul(std).add_(mu)
    
BATCH_SIZE = 29492

data_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=False)

def train(model, loss_func, optimizer, num_epochs):
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        loop = tqdm(enumerate(data_loader))

        for batch_idx, (data, _) in enumerate(data_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_func(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
        print('Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
    
def prepare_data(rawcsv):
    return rawcsv

device = torch.device('cpu')

def main():
    print("Hello World!")
    testVAE = myVAEdef()
    print(testVAE.attr)

main()