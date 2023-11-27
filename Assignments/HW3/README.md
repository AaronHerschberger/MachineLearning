# Problem Statement and Solution Methodology
(Note that the final line of main.py, line 154, is commented out as the inference function broke last minute.)
The goal of this VAE is to draw the even digits, 0,2,4,6,8, based on the training from the MNSIT dataset (after resizing it to 14 x 14).

Each image as a flattened list of 196 values, valued from 0 to 255 (representing how dark the pixel is). 
The netowrk takes these values after they have been normalized to be between [0,1].

The network has an encoding stage, a decoding stage, and backproppogation stage. The model is then used for inference, which
in this case is drawing digits 0,2,4,6,8 and exporting them to a PDF.

The encoding and decoding stage use linear regression, ReLU, and std deviation extraction to compress and decompress the
input vector, respectively. The reparameterization trick makes backproppogation possible, since normally, a sample from the 
latent vector is unable to reconstruct the input image alone and compare for loss.

In this implementation, reparameterization takes place within (at the end) of the forward() pass function.

## More details about each stage can be found in the docstring of each function in main.py.

# Run instructions
To run with default hyperparameters,

## python main.py -o result_dir -n 100


To run a verbose version of the program, which prints loss at every training epoch, type:

python main.py --v
--------------------------------

For more info, type:

python main.py --help
--------------------------------

By default, the code should generate the folder result_dir if it does not already exist and
save all resulting figures in PDF files in this folder as follows:

1) loss.pdf

This figure shows the progress of the VAE in training mode. We expect the loss
to be decreasing and converging in this figure and avoid over-fitting.

2) 1.pdf, 2.pdf, ..., 100.pdf

After training, a hundred digit sample images are made and stored
them in files of the format i.pdf where n runs from 1 to the input of flag -n.
By default, n is 100.


# Use of A.I. statement
This assignment was written with the assistance of GitHub Copilot.

# Dependancies
This project requires the following packages and class:

PyTorch (https://pytorch.org/)
torchvision (https://pypi.org/project/torchvision/)
tqdm (https://pypi.org/project/tqdm/)
argparse (https://pypi.org/project/argparse/)
VAEclass.py (in this repo)

The packages can be installed using pip install [package_name]
