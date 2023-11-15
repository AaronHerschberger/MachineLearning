import random
import torch
import torch.nn as nn

def generate_data(seed, train_size, test_size):
    random.seed(seed)
    train_data = []
    test_data = []

    for i in range(train_size + test_size):
        # Generate two random 8-bit binary numbers A and B
        A = [random.randint(0, 1) for _ in range(8)]
        B = [random.randint(0, 1) for _ in range(8)]

        # Compute the product C = A * B
        A_int = int(''.join(map(str, A[::-1])), 2)
        B_int = int(''.join(map(str, B[::-1])), 2)
        C_int = A_int * B_int
        C = list(map(int, bin(C_int)[2:].zfill(16)))[::-1]

        if i < train_size:
            train_data.append((A, B, C))
        else:
            test_data.append((A, B, C))

    return train_data, test_data

class BinaryMultiplicationRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryMultiplicationRNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)