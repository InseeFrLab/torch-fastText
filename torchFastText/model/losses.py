import torch.nn.functional as F
from torch import nn


class OneVsAllLoss(nn.Module):
    def __init__(self):
        super(OneVsAllLoss, self).__init__()

    def forward(self, logits, targets):
        """
        Compute One-vs-All loss

        Args:
            logits: Tensor of shape (batch_size, num_classes) containing classification scores
            targets: Tensor of shape (batch_size) containing true class indices

        Returns:
            loss: Mean loss value across the batch
        """

        num_classes = logits.size(1)

        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()

        # For each sample, treat the true class as positive and all others as negative
        # Using binary cross entropy for each class
        loss = F.binary_cross_entropy_with_logits(
            logits,  # Raw logits
            targets_one_hot,  # Target probabilities
            reduction="none",  # Don't reduce yet to allow for custom weighting if needed
        )

        # Sum losses across all classes for each sample, then take mean across batch
        return loss.sum(dim=1).mean()
