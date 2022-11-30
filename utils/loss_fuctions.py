import numpy as np
import math
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

class MSE:
    def __call__(self, data_dict):
        pred = data_dict['logits']
        target = data_dict['target']
        mse = F.mse_loss(pred, target)
        return mse

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def __call__(self, data_dict):

        Ii = data_dict['target']
        Ji = data_dict['logits']

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win], device=Ii.device, requires_grad=False)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        return -torch.mean(cc)


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1'):
        self.penalty = penalty

    def __call__(self, data_dict):
        logits = data_dict['logits']
        ndims = len(list(logits.size())) - 2
        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions to put the ith dimension first
            r = [d, *range(d), *range(d+1, ndims+2)]
            li = torch.permute(logits, r)
            dfi = li[1:, ...] - li[:-1, ...]

            # permute back
            r = [*range(1, d + 1), 0, *range(d + 1, ndims + 2)]
            dfi = torch.permute(dfi, r)
            df[i] = torch.abs(dfi)

        if self.penalty == 'l2':
            df = [f * f for f in df]

        grad = torch.stack([torch.mean(f) for f in df]).mean()

        return grad