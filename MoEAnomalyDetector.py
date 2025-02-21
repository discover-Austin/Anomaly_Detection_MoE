import torch
import torch.nn as nn
import torch.nn.functional as F
from ExpertModule import ForecastingExpert

class MoEAnomalyDetector(nn.Module):
    def __init__(self, num_experts: int, input_length: int, hidden_units: int):
        super(MoEAnomalyDetector, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            ForecastingExpert(input_length, hidden_units) for _ in range(num_experts)
        ])   
        self.gating_network = nn.Linear(input_length, num_experts)
    
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # Compute gating scores and convert to probabilities with softmax
        gating_scores = self.gating_network(x)  # Shape: (batch, num_experts)
        gating_probs = F.softmax(gating_scores, dim=1)    
        expert_preds = [expert(x) for expert in self.experts]
        expert_preds = torch.stack(expert_preds, dim=1)     
        gating_probs_expanded = gating_probs.unsqueeze(2)
        final_forecast = torch.sum(expert_preds * gating_probs_expanded, dim=1)  
        return final_forecast, gating_probs