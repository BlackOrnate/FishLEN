import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.morphology import skeletonize
from scipy.spatial import ConvexHull, distance_matrix


def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice


def loss_fn(model_type, pred, target):
    if model_type == "len_pred" or model_type == "len_pred_new":
        smooth_l1_loss = nn.SmoothL1Loss()(pred, target)

        return smooth_l1_loss

    elif model_type == "mask_pred":
        bce_loss = nn.BCELoss()(pred.to(torch.float), target.to(torch.float))
        dice = dice_loss(pred.to(torch.float), target.to(torch.float))

        return bce_loss + dice
