# Problem Statement and Solution Methodology
The goal of this VAE is to draw the even digits, 0,2,4,6,8, based on the training from the MNSIT dataset (after resizing it to 14 x 14).

Each image as a flattened list of 196 values, valued from 0 to 255 (representing how dark the pixel is). 
The netowrk takes these values after they have been normalized to be between [0,1].

The network has an encoding stage, a decoding stage, a backproppogation stage.

# Run instructions
To run with default hyperparameters,
--------------------------------
python main.py -o result_dir -n 100
--------------------------------

To run a verbose version of the program, which prints loss at every training epoch, type:
--------------------------------
python main.py --v
--------------------------------

For more info on params, type:
--------------------------------
python main.py --help
--------------------------------

By default, the code should generate the folder result_dir if it does not already exist and
save all result files in this folder. The result files are all pdf figures as
follows:

1) loss.pdf

This figure shows the progress of the VAE in training mode. We expect the loss
to be decreasing and converging in this figure and avoid over-fitting.

2) 1.pdf, 2.pdf, ..., 100.pdf

After training, a hundred digit sample images are made and stored
them in files of the format i.pdf where n runs from 1 to the input of flag -n.
By default, n is 100.


# Use of A.I. statement
This assignment was written with the assistance of GitHub Copilot.
