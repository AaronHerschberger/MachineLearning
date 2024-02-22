import random
import torch
import torch.nn as nn
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


def generate_data(seed, train_size, test_size, swap=False):
    # Generate a dataset for binary multiplication
    # Takes seed, for the random number generator, and
    #       train_size, for the number of training samples to generate
    #       test_size, for the number of test samples to generate
    # Returns train_AB, listof(listof(16 ints)), interleaved binary numbers A and B
    #         train_C, listof(16 ints), binary numbers C = A * B padded with 0
    #         test_AB, and
    #         test_C, the test set in the same format
    random.seed(seed)
    train_AB = []
    train_C = []
    test_AB= []
    test_C = []

    for i in range(train_size + test_size):
        # Generate two random 8-bit binary numbers A and B
        if not swap:
            B = [random.randint(0, 1) for _ in range(8)]
            A = [random.randint(0, 1) for _ in range(8)]
        else:
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

def train_model(AB_train, C_train, AB_test, C_test, model_params, optim_params):
    # Train the RNN on the dataset.
    """
    Takes interleaved AB and result C for both training and test data
    Takes model_params for hyperparameters, namely hidden layer size and epochs.
    Takes optim_params to initialize optimizer
    
    Returns the model, and 
    trLosses, tstLosses, epochNum which are all listof(Num) for graphing.
    """
    
    input_size = 16
    hidden_size = model_params['units']
    output_size = 16
    
    trLosses = []
    tstLosses = []
    epochNum = []

    model = BinaryMultiplicationRNN(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=optim_params['lr'])

    for epoch in range(model_params['epochs']):
        optimizer.zero_grad()
        hidden = model.initHidden()

        # Run over training set and compute loss
        for i in range(len(AB_train)):
            input = torch.tensor(AB_train[i], dtype=torch.float32)
            target = torch.tensor(C_train[i], dtype=torch.float32)

            output = model.forward(input)
            target = target.view(1, -1, 16)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Compute the loss on the test set
        for i in range(len(AB_test)):
            input = torch.tensor(AB_test[i], dtype=torch.float32)
            target = torch.tensor(C_test[i], dtype=torch.float32)

            output = model.forward(input)
            target = target.view(1, -1, 16)
            test_loss = 1-torch.sigmoid(criterion(output, target))
            
        if epoch % 10 == 0:
            print('Epoch: %d, Training Loss: %.5f, Test Loss: %5f' % (epoch, loss, test_loss))
            trLosses.append(float(loss))
            tstLosses.append(float(test_loss))
            epochNum.append(int(epoch))

    return model, trLosses, tstLosses, epochNum

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

        self.inputToHidden = nn.RNN(input_size, hidden_size, batch_first = True)
        self.hiddenToOut = nn.Linear(hidden_size, output_size)
    
    
    def forward(self, input):
        # Step the RNN forward one step. Takes input of size (1, input_size)
        input = input.view(1, -1, self.input_size)
        hidden, _ = self.inputToHidden(input)
        output = self.hiddenToOut(hidden)
        output = torch.sigmoid(output)
        return output

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def main():
    
    parser = argparse.ArgumentParser(description='Trains an RNN to perform multiplication of binary integers A * B = C')
    parser.format_help()
    
    parser.add_argument('--param', type=str, help='file containing hyperparameters')
    parser.add_argument('--train-size', type=int, help='size of the generated training set', default=10)
    parser.add_argument('--test-size', type=int, help='size of the generated test set', default=10)
    parser.add_argument('--seed', type=int, help='random seed used for creating the datasets', default=63)
    
    args = parser.parse_args()
    
    if args.param is None:
        print('Please type --help for help with command line arguments')
        return
    else:
        with open(args.param, 'r') as f:
            params = json.load(f)

    train_data, train_target, test_data, test_target = generate_data(args.seed, args.train_size, args.test_size)

    model, trainingLosses, testingLosses, epochNumber = train_model(train_data, train_target, test_data, test_target, params['model'], params['optim'])
    
    # Plot the training and testing losses
    plt.plot(epochNumber, trainingLosses, label = 'Training Loss')
    plt.plot(epochNumber, testingLosses, label = 'Testing Loss')
    
    # Run swapped test by switching B and A
    print('============ Running swapped test =============\n\n')
    train_data, train_target, test_data, test_target = generate_data(args.seed, args.train_size, args.test_size, swap=True)
    swapTestedModel, trainingLosses, testingLosses, epochNumber = train_model(train_data, train_target, test_data, test_target, params['model'], params['optim'])
    
    # Plot the training and testing losses
    plt.plot(epochNumber, trainingLosses, label = 'Swapped Training Loss')
    plt.plot(epochNumber, testingLosses, label = 'Swapped Testing Loss')
    
    
if __name__ == '__main__':
    main()