import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np
from math import exp
from args_setting import args


def compute_loss(fusion, img_cat, img_1, img_2, put_type='mean',
                 balance1=0.5, balance2=0.5, balance3=1,
                 w1=0.5, w2=0.5):
    loss_content = intensity_loss(fusion, img_1, img_2, put_type, w1, w2).to(args.DEVICE)
    s_loss = structure_loss(fusion, img_cat)
    loss_grad = torch.log(torch.ones(1) + s_loss).to(args.DEVICE)
    loss_ssim = (torch.ones(1) - ssim(fusion, img_1) + torch.ones(1) - ssim(fusion, img_2)).to(args.DEVICE)

    return balance1 * loss_content + balance2 * loss_grad + balance3 * loss_ssim


def create_putative(in1, in2, put_type, w1, w2):
    if put_type == 'mean':
        iput = (in1 + in2) / 2
    elif put_type == 'left':
        iput = in1
    elif put_type == 'right':
        iput = in2
    elif put_type == 'weight':
        iput = w1 * in1 + w2 * in2
    else:
        raise EOFError('No supported type!')

    return iput


def intensity_loss(fusion, img_1, img_2, put_type, w1, w2):
    inp = create_putative(img_1, img_2, put_type, w1, w2)

    # L2 norm
    loss = torch.norm(fusion - inp, 2)

    return loss


def gradient(x):
    H, W = x.shape[2], x.shape[3]

    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


def create_structure(inputs):
    B, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]

    dx, dy = gradient(inputs)

    structure = torch.zeros(B, 4, H, W)  # Structure tensor = 2 * 2 matrix

    a_00 = dx.pow(2)
    a_01 = a_10 = dx * dy
    a_11 = dy.pow(2)

    structure[:, 0, :, :] = torch.sum(a_00, dim=1)
    structure[:, 1, :, :] = torch.sum(a_01, dim=1)
    structure[:, 2, :, :] = torch.sum(a_10, dim=1)
    structure[:, 3, :, :] = torch.sum(a_11, dim=1)

    return structure


def structure_loss(fusion, img_cat):
    st_fusion = create_structure(fusion)
    st_input = create_structure(img_cat)

    # Frobenius norm
    loss = torch.norm(st_fusion - st_input)

    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average).item()
