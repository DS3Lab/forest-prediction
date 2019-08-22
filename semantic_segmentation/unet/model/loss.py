
import torch
from torch import nn
from torch.nn import functional as F
import utils
import numpy as np


class LossBinary(nn.Module):
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    """

    def __init__(self, jaccard_weight=0):
        super().__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = (1 - self.jaccard_weight) * self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss

class DiceLoss(object):
    r"""The smoothed Jaccard index loss."""

    def __init__(self, smooth=1.0):
        self.smooth = smooth

    def __call__(self, outputs, targets):
        # apply sigmoid
        outputs = F.sigmoid(outputs)

        # flatten all other dimensions
        num = outputs.size(0)
        outputs = outputs.view(num, -1)
        targets = targets.view(num, -1)

        # apply weights
        intersection = (outputs * targets).sum(1)
        scores = 2. * (intersection + self.smooth) / (
                outputs.sum(1) + targets.sum(1) + self.smooth)

        return 1 - torch.clamp(scores.sum() / num, 0.0, 1.0)


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())
