"""
This module specifies the losses we want to optimize in the neural network.
"""
import torch
from torch import nn
from torch.nn import functional as F
import utils
import numpy as np


class LossBinary(nn.Module):
    """
    Loss defined as \alpha BCE - (1 - \alpha) SoftJaccard
    Code borrowed from https://github.com/ternaus/robot-surgery-segmentation/blob/master/loss.py
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
