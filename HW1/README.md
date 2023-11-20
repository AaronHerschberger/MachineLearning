# Problem Description
This program trains an RNN to perform multiplication between two 8-digit binary numbers A and B
such that their product is C.

The binary numbers are encoded in the files in “little endian” format.

First, a python function will generate a large dataset. We assume that the factors have at most 8
digits in their binary representation.

It takes the random seed, training set size, and test set size as
arguments to the script.

When performing the multiplication, the RNN is fed an interleaved input bit string,

a_0 b_0 a_1 b_1 a_2 b_2 ... a_n b_n 0

and get a corresponding output bit string,

[padding with zeroes up to 16 digits] [digits of c]

in order to keep the input and output sequences the same length.

The program reports the training and test losses periodically during training, and
one final time when the training terminates. 

Given that multiplication is commutative, it will also compute and display the loss on 
both the training and test datasets obtained when we swap inputs A and B.


# Cmd line run instructions

Run using this command:

=========================================

python main.py --param=./param/param.json

=========================================



For parameter changes and usage, use:

=========================================

python main.py --help

=========================================


# Dependencies

This project requires the following Python packages:

- random
- numpy
- torch
- matplotlib
- argparse
- json

You can install these packages using 
-----------------
pip install torch
-----------------


# Citation

This project was developed with the assistance of GitHub Copilot