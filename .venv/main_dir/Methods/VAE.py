import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        #initialize the three layers, input - hidden - latent
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        #activation on the first -> hidden layer only then pass to mu and var
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        #initialize three layers latent - hidden - output
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        #activation on the latent -> hidden dim only, then pass to output layer
        h = torch.relu(self.fc1(z))
        #this is the reconstructed value
        # x_hat = self.sigmoid(self.fc2(h))
        x_hat = self.fc2(h)
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        #initialize system
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        # obtain parameters of the system
        mu, logvar = self.encoder(x)
        std = torch.exp(0.5 * logvar)
        #I think this is just a filler statement and can be removed as it is just adding random noise
        eps = torch.randn_like(std)
        #obtain reconstructed value from the latent space representation of the input
        z = mu + eps * std
        x_hat = self.decoder(z)
        return x_hat, mu, logvar