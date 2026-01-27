import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class AlphaZeroLoss(nn.Module):
    """AlphaZero loss: L = Σ CE(π, π̂) + (V - z)² + regularization"""
    
    def __init__(self, 
                 value_loss_weight: float = 1.0,
                 entropy_weight: float = 0.01,
                 l2_weight: float = 1e-4):
        super().__init__()
        
        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight
        self.l2_weight = l2_weight
    
    def forward(self, 
                policy_logits_list: List[torch.Tensor],
                target_policies_list: List[torch.Tensor],
                predicted_values: torch.Tensor,
                target_values: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Compute AlphaZero loss."""
        # Policy loss
        policy_loss = 0.0
        total_entropy = 0.0
        
        for logits, target in zip(policy_logits_list, target_policies_list):
            log_probs = F.log_softmax(logits, dim=-1)
            ce = -(target * log_probs).sum(dim=-1).mean()
            policy_loss += ce
            
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
            total_entropy += entropy
        
        policy_loss = policy_loss / len(policy_logits_list)
        avg_entropy = total_entropy / len(policy_logits_list)
        
        # Value loss
        value_loss = F.mse_loss(predicted_values, target_values)
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_loss_weight * value_loss - 
                     self.entropy_weight * avg_entropy)
        
        stats = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': avg_entropy.item()
        }
        
        return total_loss, stats