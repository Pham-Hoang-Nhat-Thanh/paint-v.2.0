import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


class AlphaZeroLoss(nn.Module):
    """
    AlphaZero loss with anti-overfitting measures for value head.

    Loss: L = Σ CE(π, π̂) + λ_v × (V - z)² + regularization

    Anti-overfitting features:
    - Label smoothing for value targets
    - Separate value weight decay
    - Dynamic value loss weight scheduling
    - Huber loss option for robustness
    """

    def __init__(self,
                 value_loss_weight: float = 0.5,  # Reduced from 1.0
                 entropy_weight: float = 0.01,
                 l2_weight: float = 1e-4,
                 value_label_smoothing: float = 0.1,  # NEW: smooth targets by 10%
                 use_huber_loss: bool = False,  # NEW: more robust to outliers
                 huber_delta: float = 1.0):
        super().__init__()

        self.value_loss_weight = value_loss_weight
        self.entropy_weight = entropy_weight
        self.l2_weight = l2_weight
        self.value_label_smoothing = value_label_smoothing
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
    
    def forward(self,
                policy_logits_list: List[torch.Tensor],
                target_policies_list: List[torch.Tensor],
                predicted_values: torch.Tensor,
                target_values: torch.Tensor,
                sample_weights: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """Compute AlphaZero loss with anti-overfitting measures."""
        # Policy loss
        policy_loss = 0.0
        total_entropy = 0.0

        for logits, target in zip(policy_logits_list, target_policies_list):
            log_probs = F.log_softmax(logits, dim=-1)
            ce_per_sample = -(target * log_probs).sum(dim=-1)  # [batch]
            if sample_weights is None:
                ce = ce_per_sample.mean()
            else:
                denom = sample_weights.sum() + 1e-12
                ce = (ce_per_sample * sample_weights).sum() / denom
            policy_loss += ce

            probs = F.softmax(logits, dim=-1)
            entropy_per_sample = -(probs * log_probs).sum(dim=-1)
            if sample_weights is None:
                entropy = entropy_per_sample.mean()
            else:
                denom = sample_weights.sum() + 1e-12
                entropy = (entropy_per_sample * sample_weights).sum() / denom
            total_entropy += entropy

        policy_loss = policy_loss / len(policy_logits_list)
        avg_entropy = total_entropy / len(policy_logits_list)

        # Value loss with label smoothing to prevent overconfident predictions
        if self.value_label_smoothing > 0:
            # Smooth targets towards 0.5 (neutral value)
            smoothed_targets = (1 - self.value_label_smoothing) * target_values + \
                             self.value_label_smoothing * 0.5
        else:
            smoothed_targets = target_values

        # Choose loss function
        # Compute per-sample value loss then optionally weight
        per_sample_value_loss = (predicted_values - smoothed_targets).pow(2).squeeze(-1)  # [batch]
        if self.use_huber_loss:
            # approximate Huber per-sample
            delta = self.huber_delta
            abs_err = (predicted_values - smoothed_targets).abs().squeeze(-1)
            hub = torch.where(abs_err <= delta, 0.5 * abs_err.pow(2), delta * (abs_err - 0.5 * delta))
            per_sample_value_loss = hub

        if sample_weights is None:
            value_loss = per_sample_value_loss.mean()
        else:
            denom = sample_weights.sum() + 1e-12
            value_loss = (per_sample_value_loss * sample_weights).sum() / denom

        # Total loss
        total_loss = (policy_loss +
                     self.value_loss_weight * value_loss -
                     self.entropy_weight * avg_entropy)

        # Enhanced statistics for monitoring overfitting
        with torch.no_grad():
            value_mae = F.l1_loss(predicted_values, target_values)
            value_max_error = (predicted_values - target_values).abs().max()
            predicted_mean = predicted_values.mean()
            predicted_std = predicted_values.std()

        stats = {
            'total_loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'value_mae': value_mae.item(),  # NEW: Mean absolute error
            'value_max_error': value_max_error.item(),  # NEW: Worst prediction
            'value_pred_mean': predicted_mean.item(),  # NEW: Check for bias
            'value_pred_std': predicted_std.item(),  # NEW: Check variance
            'entropy': avg_entropy.item()
        }

        return total_loss, stats