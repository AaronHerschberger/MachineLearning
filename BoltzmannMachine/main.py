import torch
import argparse
import matplotlib.pyplot as plt
from torch.autograd import Variable
import random as rand
import numpy as np

def MCMC(out_size,N,J, beta = 1.0, verbose = False):
    # Initialize the state with random spins
    init_state = rand.choice([-1,1], size = N)
    if verbose:
        print('J:',J)
        print('init state: ',init_state)

    accepted = np.zeros(N)
    # Run the MCMC until we have enough accepted states
    while (accepted.size/N) <= out_size:
        # Select a random spin to flip
        index = rand.randint(N)

        # Calculate the change in energy if we flip this spin
        n = init_state[index]
        nleft = init_state[(index-1)%N]
        jleft = J[(index-1)%N]
        nright = init_state[(index+1)%N]
        jright = J[index]
        dE = deltaE(n,jleft,jright,nleft,nright)

        # If the energy decreases, accept the new state
        if dE < 0:
            new_state = init_state
            new_state[index] = -1
            accepted = np.concatenate((accepted,new_state))

            if verbose:
                print('New state: {} dE: {}'.format(new_state,dE))
        # If the energy increases, accept the new state with a certain probability
        elif rand.random() < np.exp(-beta*dE):
            new_state = init_state
            new_state[index] *= -1

            accepted = np.concatenate((accepted,new_state))
            if verbose:
                print('New state: {} dE: {}'.format(new_state,dE))
        else:
            pass
    return accepted.reshape((-1,N))[1:]

def deltaE(spin, jl, spinl, jr, spinr):
    # Calculate the change in energy if we flip a spin
    return 2 * spin * (jl * spinl + jr * spinr)

class BoltzmannMachine:
    def __init__(self, size):
        # Initialize the weights randomly
        self.weights = Variable(torch.randn(size, size), requires_grad=True)

    def energy(self, spins):
        # Calculate the energy of a state
        return -torch.sum(self.weights * spins)

    def update(self, spins, learning_rate):
        # Calculate the energy of the state
        energy = self.energy(spins)
        # Compute the gradients
        energy.backward()
        with torch.no_grad():
            # Update the weights using gradient descent
            self.weights -= learning_rate * self.weights.grad
            # Reset the gradients to zero
            self.weights.grad.zero_()

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    # Convert the data to a list of spin configurations
    data = [[-1 if spin == '-' else 1 for spin in line.strip()] for line in lines]
    return data

def train(model, data, learning_rate, epochs):
    losses = []
    for epoch in range(epochs):
        loss = 0
        for spins in data:
            # Convert the spins to a PyTorch tensor
            spins = torch.tensor(spins, dtype=torch.float32)
            # Calculate the energy of the state
            energy = model.energy(spins)
            # Accumulate the loss
            loss += energy
        # Compute the gradients
        loss.backward()
        with torch.no_grad():
            # Update the weights using gradient descent
            model.weights -= learning_rate * model.weights.grad
            # Reset the gradients to zero
            model.weights.grad.zero_()
        # Record the loss for this epoch
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
    return {row: model.weights for row in spins}

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