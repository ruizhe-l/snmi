import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import process_methods as P

# weight maps ----------------------------------------------

def balance_weight_map(target, epsilon=1e-9):
    axis = tuple(range(2, target.ndim))
    ccount = target.sum(axis, keepdim=True)
    c = 1/(torch.sum(1/(epsilon + ccount), axis=1, keepdim=True))
    weight = c / ccount
    weight_map = torch.tile(weight, [1,1] + list(target.shape[2:]))
    return weight_map

def feedback_weight_map(logits, target, alpha=3, beta=100):
    probs = F.softmax(logits, dim=1)
    p = torch.sum(probs * target, axis=1, keepdim=True)
    weight_map = torch.exp(-torch.pow(p, beta)*np.log(alpha))
    weight_map = torch.tile(weight_map, [1] + [logits.shape[1]] + [1]*(logits.ndim-2))
    return weight_map 

# ----------------------------------------------------------


def weighted_cross_entropy(logits, target, weight):
    prob = F.softmax(logits, dim=1)
    loss = -torch.sum(target * torch.log(prob) * weight, axis=[1])
    return loss

class CrossEntropy:
    def __init__(self, **kwargs):
        self.criterion = nn.CrossEntropyLoss(**kwargs)

    def __call__(self, data_dict):
        logits = data_dict['logits']
        target = data_dict['target']
        target = torch.argmax(target, dim=1)
        loss = self.criterion(logits, target)
        return loss


class BalanceCrossEntropy:
    def __call__(self, data_dict):
        logits = data_dict['logits']
        target = data_dict['target']
        weight_map = balance_weight_map(target)
        loss_map =  weighted_cross_entropy(logits, target, weight_map)
        loss = loss_map.mean()
        return loss

class FeedbackCrossEntropy:
    def __init__(self, alpha=3, beta=100):
        self.alpha = alpha
        self.beta = beta

    def __call__(self, data_dict):
        logits = data_dict['logits']
        target = data_dict['target']
        weight_map = feedback_weight_map(logits, target, self.alpha, self.beta)
        loss_map = weighted_cross_entropy(logits, target, weight_map)
        loss = loss_map.mean()
        return loss

class SoftDice():
    def __init__(self, ignore_background=True):
        self.ignore_background = ignore_background

    def __call__(self, data_dict):
        logits = data_dict['logits']
        target = data_dict['target']
        pred = F.softmax(logits, dim=1)
        axis = tuple(range(2, target.ndim))
        intersection = torch.sum(pred * target, axis)
        sum_ = torch.sum(pred + target, axis)
        soft_dice = 2 * intersection / sum_
        if self.ignore_background:
            soft_dice = soft_dice[:, 1:]
        return 1 - soft_dice.mean()
