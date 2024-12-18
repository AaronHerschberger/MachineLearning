import torch
import torch.nn as nn
import torch.nn.functional as funcs
from torch.utils.data import DataLoader  

import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from VAEclass import VAE


def prepare_data(list_from_csv):
    """Converts listof(str) from csv file to listof(listof(floats)), then 
    normalizes each item on each line (that's why there are 2 map() functions).)"""
    
    # First, split whole file into lines; then split lines into str, convert to 
    # float, and normalize to [0,1] by dividing by 255. In future, 2 for loops 
    # are much easier to implement and read than map and lambda.
    normalized = map(lambda line: line.split(), list_from_csv)
    normalized = map(lambda line: map(lambda num: float(num)/255, line), list(normalized))
    return list(map(list, normalized))

# Declare global variables
BATCH_SIZE = 29492
INPUT_DIM = 197 # 14*14 = 196 + 1 for label
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

def train(model, loss_func, optimizer, num_epochs=NUM_EPOCHS, verbose=True):
    """
    Trains the model for num_epochs epochs. 
    
    Takes an empty VAE model, a loss function, an optimizer, and the number of epochs to train for.
    
    Returns a list containing the loss values at each epoch.
    """
    # Start training
    for epoch in range(num_epochs):
        # loop is the input data; tqdm is a progress bar for visualization.
        
        lossList = []
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
            # loop.set_postfix(loss=loss.item())
        
        if True:
            print('Epoch: {}, Average loss: {:.4f}'.format(epoch+1, loss.item()))
            
        lossList.append(loss.item())
        
    return lossList
    
def inference(model, num_examples=100):
    """
    Takes a trained model, Generates num_examples of MNIST-like digits.
    Specifically we iterate over each digit,
    then since we have the mu & sigma representation for
    each digit we can sample from the distribution to generate.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    
    digit = 0
    images = []
    idx = 0
    # Get images of each even digit from 0 to 8
    for x in trainset:
        if x[-1] == idx: # label is last element
            images.append(x)
            idx += 2
        if idx == 8:
            break

    encodings_digit = []
    for d in range(2):
        with torch.no_grad():
            mu, sigma = model.encode(torch.Tensor(images[d]))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    
    # Generate num_examples of PDFs by sampling from the distribution of the original
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        # out = out.view(-1, 1, 28, 28)
        save_image(out, f"{example}.pdf")
        
        
def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Trains a VAE to generate even MNIST digits, then exports a loss plot PDF and n PDFs of generated digits.')
    parser.format_help()
    parser.add_argument('-v', '--verbose', action='store_true', help='print loss values at each epoch')
    parser.add_argument('-e', type=int, help='number of epochs to train', default=3)
    parser.add_argument('-n', type=int, help='number of generated digits to export', default=100)
    # parser.add_argument('--inputPath', type=str, help='input file', default='data/even_mnist.csv')
    args = parser.parse_args()
    
    # initialize model, optimizer, and loss function
    model = VAE(INPUT_DIM, 20, 10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    # Run training
    lossValues = train(model, loss_fn, optimizer, args.e, args.verbose)
    print(lossValues)
    
    # Make and Export loss plot
    def plot_graph(data, filename):
        plt.figure()
        plt.plot(range(len(data)), data)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Value vs Epoch')
        plt.savefig(filename, format='pdf')
    
    plot_graph(lossValues, 'loss.pdf')

    # Run inference
    # inference(model, args.n)

main()