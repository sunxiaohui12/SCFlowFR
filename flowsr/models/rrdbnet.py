"""
RRDBNet for Latent Space Super-Resolution
Adapted from ELIR project (https://github.com/eladc-git/ELIR)
"""

import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Block2D(nn.Module):
    def __init__(self, c_in, c_out, kernel=3, activation=True, overparametrization=False):
        super().__init__()
        if overparametrization:
            self.block = nn.Sequential(
                nn.Conv2d(c_in, 4*c_out, kernel_size=kernel, padding=1),
                nn.Conv2d(4*c_out, c_out, kernel_size=1),
                nn.SiLU() if activation else nn.Identity()
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=kernel, padding=1),
                nn.SiLU() if activation else nn.Identity()
            )

    def forward(self, x):
        return self.block(x)


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, c_in=4, c_hid=32, overparametrization=False):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = Block2D(c_in, c_hid, 3, True, overparametrization)
        self.conv2 = Block2D(c_in + c_hid, c_hid, 3, True, overparametrization)
        self.conv3 = Block2D(c_in + 2 * c_hid, c_hid, 3, True, overparametrization)
        self.conv4 = Block2D(c_in + 3 * c_hid, c_hid, 3, True, overparametrization)
        self.conv5 = Block2D(c_in + 4 * c_hid, c_in, 3, False, overparametrization)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, c_inout=4, c_hid=64, overparametrization=False):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(c_inout, c_hid, overparametrization)
        self.RDB2 = ResidualDenseBlock_5C(c_inout, c_hid, overparametrization)
        self.RDB3 = ResidualDenseBlock_5C(c_inout, c_hid, overparametrization)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDBNet: Latent Reconstruction Module for FlowScale
    
    Adapted from ELIR project for latent space super-resolution.
    Used to generate context in latent space for flow matching.
    
    Args:
        c_inout: Input/output channel number (latent space channels, default 4 for TAESD)
        c_hid: Hidden channel number
        n_rrdb: Number of RRDB blocks
        overparametrization: Whether to use overparameterization (improves training stability)
    """
    def __init__(self, c_inout=4, c_hid=64, n_rrdb=3, overparametrization=True):
        super(RRDBNet, self).__init__()
        self.overparametrization = overparametrization
        layers = []
        for _ in range(n_rrdb):
            layers.append(RRDB(c_inout, c_hid, overparametrization))
        self.lrm = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.lrm(x)









