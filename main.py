import argparse
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
import csv

from VAEclass import myVAEdef


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        
        self.hid_2mu = nn.Linear(200, 2)
        self.hid_2sigma = nn.Linear(200, 2)
        
        # Encode function
        self.in2hidden = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20 + 20),
            nn.ReLU(),
            nn.Linear(20 + 20, 20 + 20),
            nn.ReLU()
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

        mu = self.hid_2mu(hidden)
        sigma = self.hid_2sigma(hidden)
        return mu, sigma
    
    def decode(self, z):
        hidden = self.latent2hidden(z)
        output = funcs.sigmoid(hidden)
        
        return output
    
    def forward(self, x):
        mu_logvar = self.endode(x.view(-1, INPUT_DIM))
        mu = mu_logvar[:, :20]
        logvar = mu_logvar[:, 20:]
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    # def reparameterize(self, mu, logvar):
    #     std = torch.exp(0.5*logvar)
    #     eps = torch.rand_like(std)
    #     return eps.mul(std).add_(mu)
    

def prepare_data(list_from_csv):
    """Converts listof(str) from csv file to listof(listof(floats)), then 
    normalizes each item on each line (that's why there are 2 map() functions).)"""
    
    # First, split whole file into lines; then split lines into str, convert to 
    # float, and normalize to [0,1] by dividing by 255. In future, 2 for loops 
    # are much better than map and lambda.
    normalized = map(lambda line: line.split(), list_from_csv)
    normalized = map(lambda line: map(lambda num: float(num)/255, line), list(normalized))
    return list(map(list, normalized))


def train(model, loss_func, optimizer, num_epochs, verbose=False):
    """
    Trains the model for num_epochs epochs. 
    """
    # Start training
    for epoch in range(num_epochs):
        # loop is the input data; tqdm is a progress bar for visualization.
        # loop = tqdm(enumerate(data_loader))
        
        loop = trainset
        for line in loop:
            # Forward pass
            line = torch.Tensor(line).to(device).view(-1, INPUT_DIM)
            # Update z, mu, and sigma
            x_reconst, mu, sigma = model(line)

            # Calculate loss and KL divergence
            reconst_loss = loss_func(x_reconst, line)
            kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            # Backprop and optimize
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
        print('Epoch: {} Average loss: {:.4f}'.format(epoch, loss.item()))
    
def inference(model, digit, num_examples=1):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    images = []
    idx = 0
    for x, y in trainset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 2:
            break

    encodings_digit = []
    for d in range(2):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, f"generated_{digit}_ex{example}.png")
        

BATCH_SIZE = 29492
INPUT_DIM = 197
LR_RATE = 0.001
NUM_EPOCHS = 10
inputFile = 'data/even_mnist.csv'

trainset = list(open(inputFile, 'r'))
trainset = prepare_data(trainset)
data_loader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=False)
device = torch.device('cpu')

myTransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def main():
    parser = argparse.ArgumentParser(description='Trains a VAE to generate even MNIST digits, then exports a loss plot PDF and n PDFs of generated digits.')
    parser.format_help()
    parser.add_argument('-v', '--verbose', action='store_true', help='print loss values at each epoch')
    parser.add_argument('-e', type=int, help='number of epochs to train', default=10)
    parser.add_argument('-n', type=int, help='number of generated digits to export', default=20)
    # parser.add_argument('--inputPath', type=str, help='input file', default='data/even_mnist.csv')
    
    args = parser.parse_args()
    # global inputFile
    # inputFile = args.inputPath
    
    

    # initialize model, optimizer, and loss function
    model = VAE(INPUT_DIM, 50, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    # Run training
    train(model, loss_fn, optimizer, NUM_EPOCHS, args.verbose)

main()