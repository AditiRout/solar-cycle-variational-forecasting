import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNetFeatureExtractor(nn.Module):
    """
    1D Dilated Causal Convolutional Network for multi-scale temporal modeling.
    """
    def __init__(self, in_channels=1, residual_channels=64, num_layers=8):
        super(WaveNetFeatureExtractor, self).__init__()
        self.conv_in = nn.Conv1d(in_channels, residual_channels, kernel_size=1)
        self.dilated_layers = nn.ModuleList()
        
        for i in range(num_layers):
            dilation = 2 ** i
            self.dilated_layers.append(
                nn.Conv1d(
                    residual_channels, 
                    residual_channels, 
                    kernel_size=3, 
                    padding=dilation, 
                    dilation=dilation
                )
            )
            
    def forward(self, x):
        # x input shape: (batch_size, seq_len, in_channels)
        x = x.transpose(1, 2) # Change to (batch, channels, seq_len)
        x = self.conv_in(x)
        
        for layer in self.dilated_layers:
            residual = x
            x = F.relu(layer(x)) + residual # Gated residual connection
            
        return x.transpose(1, 2) # Return back as (batch, seq_len, channels)

class WaveNetVRNN(nn.Module):
    """
    Hybrid WaveNet + Variational Recurrent Neural Network with KL Divergence.
    """
    def __init__(self, input_dim=1, latent_dim=32, hidden_dim=128, forecast_len=132):
        super(WaveNetVRNN, self).__init__()
        self.forecast_len = forecast_len
        self.feature_extractor = WaveNetFeatureExtractor(in_channels=input_dim)
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        
        # Latent Space Parameters (Mu and LogVar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder / Forecast Generator
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, forecast_len * 2) # Outputs Mean (mu) and Variance (sigma)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        features = self.feature_extractor(x)
        _, (h_n, _) = self.lstm(features)
        h_last = h_n[-1]
        
        mu_z = self.fc_mu(h_last)
        logvar_z = self.fc_logvar(h_last)
        
        z = self.reparameterize(mu_z, logvar_z)
        
        decoded = self.decoder(z)
        pred_mu = decoded[:, :self.forecast_len].unsqueeze(-1)
        pred_logvar = decoded[:, self.forecast_len:].unsqueeze(-1)
        
        return pred_mu, pred_logvar, mu_z, logvar_z

def elbo_loss(pred_mu, pred_logvar, target, z_mu, z_logvar, beta=0.01):
    """
    Evidence Lower Bound (ELBO) Loss = Reconstruction Loss + Beta * KL Divergence
    """
    # Reconstruction Loss (Gaussian Negative Log Likelihood / MSE)
    recon_loss = F.mse_loss(pred_mu, target)
    
    # KL Divergence Penalty: KL( N(mu, sigma) || N(0, I) )
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_loss = kl_loss / target.size(0)
    
    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss