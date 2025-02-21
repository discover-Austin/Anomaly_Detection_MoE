import torch
import torch.nn as nn
import torch.optim as optim
from MoEAnomalyDetector import MoEAnomalyDetector

def train_model(model, optimizer, criterion, data_loader, epochs: int = 50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for sequences, targets in data_loader:
            optimizer.zero_grad()
            # sequences shape: (batch, input_length)
            forecasts, _ = model(sequences)
            loss = criterion(forecasts, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * sequences.size(0)
        avg_loss = total_loss / len(data_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")