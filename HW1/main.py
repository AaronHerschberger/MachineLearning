import random
import os
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
import os
import matplotlib.pyplot as plt


def generate_data(seed, train_size, test_size):
    random.seed(seed)
    train_AB = []
    train_C = []
    test_AB= []
    test_C = []

    for i in range(train_size + test_size):
        # Generate two random 8-bit binary numbers A and B
        A = [random.randint(0, 1) for _ in range(8)]
        B = [random.randint(0, 1) for _ in range(8)]
        # Interleave bits of A and B to obtain the input
        AB_braided = [bit for pair in zip(A, B) for bit in pair]

        # Compute the product C = A * B
        A_int = int(''.join(map(str, A[::-1])), 2)
        B_int = int(''.join(map(str, B[::-1])), 2)
        C_int = A_int * B_int
        C = list(map(int, bin(C_int)[2:].zfill(16)))[::-1]

        if i < train_size:
            train_AB.append(AB_braided)
            train_C.append(C)
        else:
            test_AB.append(AB_braided)
            test_C.append(C)

    return train_AB, train_C, test_AB, test_C


import torch.optim as optim

def train_model(X_train, Y_train, model_params, optim_params):
    # Train the RNN on the dataset
    input_size = 16
    hidden_size = model_params['units']
    output_size = 16
    
    lossTracker = []

    model = BinaryMultiplicationRNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=optim_params['lr'])

    for epoch in range(model_params['epochs']):
        optimizer.zero_grad()
        hidden = model.initHidden()

        for i in range(len(X_train)):
            input = torch.tensor(X_train[i], dtype=torch.float32)
            target = torch.tensor(Y_train[i], dtype=torch.float32)

            # output, hidden = model(input, hidden, 16)
            output = model.forward(input)
            target = target.view(1, -1, 16)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        if epoch % 10 == 0:
            print('Epoch: %d, Loss: %.4f' % (epoch, loss))
            lossTracker.append(loss)

    return model, lossTracker

### Deprecated function, no longer used
# def compute_loss(X_test, Y_test, model):
#     # Compute and display the loss on the training and test sets
#     loss = model.evaluate(X_test, Y_test)
#     print('Loss:', loss)

class BinaryMultiplicationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryMultiplicationRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # self.inputToHidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hiddenToOut = nn.Linear(hidden_size, output_size)

        self.rnn = nn.RNN(input_size, hidden_size, batch_first = True)
    
    
    def forward(self, input):
        input = input.view(1, -1, self.input_size)
        hidden, _ = self.rnn(input)
        output = self.hiddenToOut(hidden)
        output = torch.sigmoid(output)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def main():
    parser = argparse.ArgumentParser(description='Trains an RNN to perform multiplication of binary integers A * B = C')
    parser.add_argument('--param', type=str, help='file containing hyperparameters')
    parser.add_argument('--train-size', type=int, help='size of the generated training set', default=10)
    parser.add_argument('--test-size', type=int, help='size of the generated test set', default=10)
    parser.add_argument('--seed', type=int, help='random seed used for creating the datasets', default=63)
    args = parser.parse_args()

    with open(args.param, 'r') as f:
        params = json.load(f)

    train_data, train_target, test_data, test_target = generate_data(args.seed, args.train_size, args.test_size)

    model, lossOverTime = train_model(train_data, train_target, params['model'], params['optim'])
    # compute_loss(train_target, test_target, model)

if __name__ == '__main__':
    main()