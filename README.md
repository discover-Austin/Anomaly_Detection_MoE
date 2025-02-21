# Anomaly Detection in Time Series Data using a Custom Mixture of Experts

## Overview

This project tackles the detecting anomalies in time series data using a Mixture of Experts (MoE) model developed from scratch with PyTorch. Its focus is on forecasting the next value in a time series and using the prediction error as an anomaly score.

## Project Structure

- **ExpertModule.py**: Implements a simple forecasting expert. Each expert predicts the next time step from a sequence.
- **MoEAnomalyDetector.py**: Combines multiple expert modules using a gating network that learns to weight expert predictions. The combined forecast is used to compute an anomaly score.
- **train_anomaly.py**: Contains a training routine that learns to forecast normal time series behavior.
- **main.py**: Generates synthetic time series data, trains the model, evaluates forecasts, and computes anomaly scores.

## How It Works

1. **Forecasting by Experts**:  
   Each expert is a simple feed-forward neural network that takes a window of past time steps and forecasts the next value.

2. **Gating Mechanism**:  
   A gating network assigns weights to each expert's forecast. These weighted forecasts are summed to produce the final prediction.

3. **Anomaly Score**:  
   The discrepancy (absolute error) between the predicted and actual next value serves as an anomaly score. High error indicates a potential anomaly.

## To Run

1. Install PyTorch if you haven't already:
   ```
   pip install torch
   ```
2. Run the main entry point:
   ```
   python main.py
   ```