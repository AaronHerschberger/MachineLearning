import torch.nn as nn
import torch.nn.functional as funcs
import torch

class VAE(nn.Module):
    """Variational Autoencoder (VAE) with a standard normal prior distribution.
    
    Includes a 2-layer encoder and 2-layer decoder, which are both fully connected; and a forward pass function, which performs the reparameterization trick.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.input_dim = input_dim
        
        self.hid_2mu = nn.Linear(hidden_dim, 10)
        self.hid_2sigma = nn.Linear(hidden_dim, 10)
        self.hid2out = nn.Linear(10, hidden_dim)
        
        # Encode sequence
        self.in2hidden = nn.Sequential(
            nn.Linear(input_dim, 50),
            nn.ReLU(),
            nn.Linear(50, hidden_dim),
            nn.ReLU()
        )
        
        # Decode sequence
        self.latent2hidden = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.ReLU(),
            nn.Linear(50, input_dim),
            nn.ReLU()
        )
        
    def endode(self, x):
        """Takes input vector x, outputs mu and sigma: the mean and standard deviation of the latent distribution, which is later sampled (in the "forward" function) to generate a latent vector."""
        hidden = self.in2hidden(x)

        mu = self.hid_2mu(hidden)
        sigma = self.hid_2sigma(hidden)
        return mu, sigma
    
    def decode(self, z):
        """Takes the latent vector z, as input and outputs the reconstructed input."""
        new_hidden = self.hid2out(z)
        new_hidden = self.latent2hidden(new_hidden)
        output = funcs.sigmoid(new_hidden)
        
        
        return output
    
    def forward(self, x):
        """Takes input vector x, calls encode, reparameterizes, calls decode,
        and outputs the reconstructed input, mu, and sigma."""
        mu, sigma = self.endode(x)

        # Sample from latent distribution from encoder
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu + sigma*epsilon

        x = self.decode(z_reparametrized)
        return x, mu, sigma