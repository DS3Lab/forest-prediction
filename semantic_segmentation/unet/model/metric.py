import numpy as np
import utils
from torch import nn
import torch

def get_jaccard(y_pred, y_true):
    """
    Jaccard index
    """
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)
    result = ((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy()
    return np.mean(result)
    # return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())
