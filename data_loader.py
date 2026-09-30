import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_silso(url="https://www.sidc.be/silso/INFO/snmtotcsv.php"):
    """
    Downloads SILSO monthly total sunspot numbers (v2.0), applies 13-month moving average
    smoothing, and scales features to [0, 1].
    
    Data granularity: 1 row = 1 month (1749 to present).
    """
    col_names = ["Year", "Month", "FractionalYear", "SSN", "StdDev", "Observations", "Definitive"]
    
    # Read monthly semicolon-separated SILSO CSV file
    df = pd.read_csv(url, sep=';', header=None, names=col_names)
    
    # Clean missing values (-1.0 or -1 indicator)
    df['SSN'] = df['SSN'].replace(-1.0, np.nan).replace(-1, np.nan).interpolate()
    
    # Apply standard 13-month moving average smoothing filter across monthly observations
    df['SSN_Smoothed'] = df['SSN'].rolling(window=13, center=True).mean().bfill().ffill()
    
    # Min-Max Feature Normalization [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_ssn = scaler.fit_transform(df[['SSN_Smoothed']].values)
    
    return df, scaled_ssn, scaler

def create_sequence_windows(data, input_len=528, forecast_len=132):
    """
    Splits continuous monthly time-series into sliding sequence windows.
    T = 528 months (~44 years / 4 full 11-year solar cycles)
    N = 132 months (~11 years / 1 full solar cycle)
    """
    X, Y = [], []
    for i in range(len(data) - input_len - forecast_len):
        X.append(data[i : i + input_len])
        Y.append(data[i + input_len : i + input_len + forecast_len])
    return np.array(X), np.array(Y)

class SolarDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]