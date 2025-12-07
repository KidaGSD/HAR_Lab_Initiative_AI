"""
Custom Loss Functions for HAR
Includes Focal Loss for handling class imbalance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    From "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        gamma: Focusing parameter (default: 2.0). Higher = more focus on hard examples.
        alpha: Class weights tensor (optional). If None, no class weighting.
        ignore_index: Index to ignore in loss computation (default: -1).
    """
    def __init__(self, gamma=2.0, alpha=None, ignore_index=-1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Predictions of shape (N, C) where C is num classes
            targets: Ground truth of shape (N,)
        """
        # Filter out ignored indices
        if self.ignore_index >= 0:
            mask = targets != self.ignore_index
            inputs = inputs[mask]
            targets = targets[mask]
        
        if inputs.numel() == 0:
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        # Compute softmax probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get probability of correct class
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy with label smoothing."""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
    
    def forward(self, inputs, targets):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smoothed labels
        n_classes = inputs.size(1)
        smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), confidence)
        
        # Compute loss
        loss = (-smooth_targets * log_probs).sum(dim=1)
        
        if self.weight is not None:
            weight = self.weight[targets]
            loss = loss * weight
        
        return loss.mean()
