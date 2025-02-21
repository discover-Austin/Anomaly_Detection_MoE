import torch
import torch.nn as nn
import torch.nn.functional as F
    def __init__(self, input_length: int, hidden_units: int):
        super(ForecastingExpert, self).__init__()
        self.fc1 = nn.Linear(input_length, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        prediction = self.fc2(x)
        return prediction