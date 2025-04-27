import torch
import torch.nn as nn
import numpy as np
import copy

from .convnext import *
from .layers import ConvBlock3x3, Self_Attn, ConvBlock, upsample, upsample4
from collections import OrderedDict

import torch.nn.functional as F
import scipy.io as sio
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class Equi_convnext_tea_omni(nn.Module):
    def __init__(self, equi_h, equi_w, invalid_ids=[], num_classes=13, init_bias=0.0, **kwargs):
        super(Equi_convnext_tea_omni, self).__init__()

        self.equi_encoder_sph = convnext_tiny_omni(pretrained=False)
        self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(768, 10)
        self.equi_h = equi_h
        self.equi_w = equi_w

    def forward(self, input_equi_image):
        outputs = {}

        # equi_enc_feat0,equi_enc_feat1,equi_enc_feat2,equi_enc_feat3,equi_enc_feat4,xout = self.equi_encoder(input_equi_image)
        equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph,xout_sph = self.equi_encoder_sph(input_equi_image)

        # to_save = torch.mean(equi_enc_feat2_sph,dim=2)


        out = self.fc(xout_sph)
        outputs["equi_enc_feat0"] = equi_enc_feat0_sph
        outputs["result"] = out
        outputs["equi_enc_feat1"] = equi_enc_feat1_sph
        outputs["equi_enc_feat2"] =equi_enc_feat2_sph
        outputs["equi_enc_feat3"] = equi_enc_feat3_sph
        outputs["equi_enc_feat4"] = equi_enc_feat4_sph
        return outputs

