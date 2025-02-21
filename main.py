import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from MoEAnomalyDetector import MoEAnomalyDetector
from train_anomaly import train_model

def generate_synthetic_timeseries(n_samples: int, input_length: int):
    np.random.seed(42)
    time = np.linspace(0, 100, n_samples + input_length)
    series = np.sin(time) + 0.1 * np.random.randn(n_samples + input_length)
    anomaly_indicator = np.zeros(n_samples)
    num_anomalies = int(0.05 * n_samples)
    anomaly_indices = np.random.choice(np.arange(n_samples), num_anomalies, replace=False)
    series[anomaly_indices + input_length] += np.random.choice([3, -3], size=num_anomalies)
    anomaly_indicator[anomaly_indices] = 1
    sequences = []
    targets = []
    for i in range(n_samples):
        sequences.append(series[i:i+input_length])
        targets.append(series[i+input_length])
    sequences = np.array(sequences)
    targets = np.array(targets).reshape(-1, 1)
    sequences = torch.tensor(sequences, dtype=torch.float32)
    targets = torch.tensor(targets, dtype=torch.float32)
    anomaly_indicator = torch.tensor(anomaly_indicator, dtype=torch.float32)
    return sequences, targets, anomaly_indicator

def main():
    n_samples = 1000
    input_length = 20
    sequences, targets, anomaly_indicator = generate_synthetic_timeseries(n_samples, input_length)
    dataset = TensorDataset(sequences, targets)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_experts = 3
    hidden_units = 32
    model = MoEAnomalyDetector(num_experts, input_length, hidden_units)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print("Training MoE Anomaly Detector...")
    train_model(model, optimizer, criterion, data_loader, epochs=50)
    model.eval()
    with torch.no_grad():
        forecasts, gate_probs = model(sequences)
        errors = torch.abs(forecasts - targets)
    print("\nSample Forecast Errors and Anomaly Indicators")
    for i in range(10):
        print(f"Sample {i+1}: Forecast = {forecasts[i].item():.3f}, Target = {targets[i].item():.3f}, Error = {errors[i].item():.3f}, Anomaly GT = {anomaly_indicator[i].item()}")
    
if __name__ == "__main__":
    main()
