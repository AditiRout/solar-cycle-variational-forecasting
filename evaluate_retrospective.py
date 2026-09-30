import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import load_and_preprocess_silso, create_sequence_windows, SolarDataset
from models.wavenet_vrnn import WaveNetVRNN, elbo_loss

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using compute device: {device}")
    
    print("Loading SILSO dataset...")
    df, scaled_ssn, scaler = load_and_preprocess_silso()
    
    X, Y = create_sequence_windows(scaled_ssn, input_len=528, forecast_len=132)
    train_size = int(len(X) * 0.85)
    
    train_dataset = SolarDataset(X[:train_size], Y[:train_size])
    test_X = torch.tensor(X[train_size:], dtype=torch.float32).to(device)
    test_Y = torch.tensor(Y[train_size:], dtype=torch.float32).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    model = WaveNetVRNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    print("Training WaveNet-VRNN model...")
    num_epochs = 15
    
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{num_epochs}", leave=True)
        for batch_x, batch_y in progress_bar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            pred_mu, pred_logvar, z_mu, z_logvar = model(batch_x)
            
            # Constrain logvar to prevent exploding variance
            pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=1.0)
            
            loss, recon, kl = elbo_loss(pred_mu, pred_logvar, batch_y, z_mu, z_logvar, beta=0.001)
            loss.backward()
            optimizer.step()
            
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "MSE": f"{recon.item():.4f}"})
            
    print("\nEvaluating Cycle 24 forecast with uncertainty bounds...")
    model.eval()
    with torch.no_grad():
        pred_mu, pred_logvar, _, _ = model(test_X[-1:])
        pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=1.0)
        
        # Scale back to original sunspot units
        scale_range = scaler.data_max_[0] - scaler.data_min_[0]
        pred_mu_np = scaler.inverse_transform(pred_mu.cpu().squeeze(0).numpy()).flatten()
        pred_std_np = (torch.exp(0.5 * pred_logvar).cpu().squeeze(0).numpy() * scale_range).flatten()
        actual_np = scaler.inverse_transform(test_Y[-1].cpu().numpy()).flatten()
        
        # Enforce physical constraint (SSN >= 0)
        pred_mu_np = np.clip(pred_mu_np, a_min=0, a_max=None)
        lower_bound = np.clip(pred_mu_np - 1.96 * pred_std_np, a_min=0, a_max=None)
        upper_bound = pred_mu_np + 1.96 * pred_std_np

    # Plot Retrospective Forecast
    months = np.arange(1, 133)
    plt.figure(figsize=(10, 5))
    plt.plot(months, actual_np, label="Actual Sunspot Number (Cycle 24)", color="black", linewidth=1.5)
    plt.plot(months, pred_mu_np, label="VRNN Forecast Mean (Mu)", color="blue", linewidth=2)
    plt.fill_between(
        months, 
        lower_bound, 
        upper_bound, 
        color="blue", 
        alpha=0.25, 
        label="95% Confidence Interval"
    )
    plt.title("Solar Cycle Retrospective Forecast with Probabilistic Uncertainty")
    plt.xlabel("Forecast Horizon (Months)")
    plt.ylabel("Sunspot Number (SSN)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("retrospective_forecast.png", dpi=300)
    print("Evaluation complete! Clean plot saved as 'retrospective_forecast.png'.")

if __name__ == "__main__":
    run_evaluation()