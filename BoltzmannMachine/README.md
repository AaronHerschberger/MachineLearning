# Problem Description
This project implements and trains a fully visible Boltzmann machine on
data gathered from a 1-D classical Ising chain and uses it to predict the model
couplers in absence of prior knowledge of the coupler values and only using the
model structure (1D closed chain) and the training dataset. The training
dataset is generated from a Monte-Carlo simulation of the unknown model. For
simplicity it is assumed that the thermodynamic beta is 1 throughout.


# Cmd line running instructions
The command to run the script is:
--------------------------------
python main.py data/in.txt
--------------------------------


# Dependencies

This project requires the following Python packages:

- random
- numpy
- torch
- matplotlib
- argparse

You can install these packages using 
-----------------
pip install numpy
-----------------


# Citation

This project was developed with the assistance of GitHub Copilot