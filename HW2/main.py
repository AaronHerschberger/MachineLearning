import torch
import argparse
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random as rand
import numpy as np

def MCMC(out_size,N,J, beta = 1.0, verbose = False):

        init_state = rand.choice([-1,1], size = N)
        if verbose:
            print('J:',J)
            print('init state: ',init_state)

        accepted = np.zeros(N)
        while (accepted.size/N) <= out_size:

            index = rand.randint(N)

            n = init_state[index]

            nleft = init_state[(index-1)%N]
            jleft = J[(index-1)%N]
            nright = init_state[(index+1)%N]
            jright = J[index]
            dE = deltaE(n,jleft,jright,nleft,nright)

            if dE < 0:
                new_state = init_state
                new_state[index] = -1
                accepted = np.concatenate((accepted,new_state))

                if verbose:
                    print('New state: {} dE: {}'.format(new_state,dE))
            elif rand.random() < np.exp(-betadE):
                new_state = init_state
                new_state[index] *= -1

                accepted = np.concatenate((accepted,new_state))
                if verbose:
                    print('New state: {} dE: {}'.format(new_state,dE))
            else:
                pass
        return accepted.reshape((-1,N))[1:]

def deltaE(spin, jl, spinl, jr, spinr):
        return 2 * spin * (jl * spinl + jr * spinr)

class BoltzmannMachine:
    def __init__(self, size):
        self.weights = Variable(torch.randn(size, size), requires_grad=True)

    def energy(self, spins):
        return -torch.sum(self.weights * spins)

    def update(self, spins, learning_rate):
        energy = self.energy(spins)
        energy.backward()
        with torch.no_grad():
            self.weights -= learning_rate * self.weights.grad
            self.weights.grad.zero_()

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [[-1 if spin == '-' else 1 for spin in line.strip()] for line in lines]
    return data

def train(model, data, learning_rate, epochs):
    losses = []
    for epoch in range(epochs):
        loss = 0
        for spins in data:
            spins = torch.tensor(spins, dtype=torch.float32)
            energy = model.energy(spins)
            loss += energy
        loss.backward()
        with torch.no_grad():
            model.weights -= learning_rate * model.weights.grad
            model.weights.grad.zero_()
        losses.append(loss.item())
    return losses

def predict(model, spins):
    # Convert the spins to a PyTorch tensor
    spins = torch.tensor(spins, dtype=torch.float32)
    # Disable gradient computation
    with torch.no_grad():
        # Compute the energy of the given spin configuration
        energy = model.energy(spins)
    # Return a dictionary of predicted coupler values
    print(model.weights)
    return {tuple(pair): weight.item() for pair, weight in zip(spins, model.weights)}

def main():
    # Create a parser for command line arguments
    parser = argparse.ArgumentParser()
    # Add an argument for the filename
    parser.add_argument('filename')
    # Add an optional argument for the learning rate, with a default value of 0.01
    parser.add_argument('--learning_rate', type=float, default=0.01)
    # Add an optional argument for the number of epochs, with a default value of 100
    parser.add_argument('--epochs', type=int, default=100)
    # Parse the command line arguments
    args = parser.parse_args()
    

    # Load the training data from the file
    data = load_data(args.filename)
    # Create a Boltzmann Machine model with the appropriate size
    model = BoltzmannMachine(len(data[0]))
    # Train the model and get the losses for each epoch
    losses = train(model, data, args.learning_rate, args.epochs)

    # Plot the losses and save the plot to a file
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')

    # For each spin configuration in the data, predict the coupler values and print them
    for spins in data:
        predictions = predict(model, spins)
        print(predictions)

# If this script is being run directly (not imported), call the main function
if __name__ == '__main__':
    main()