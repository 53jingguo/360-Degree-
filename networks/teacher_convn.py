import torch
import torch.nn as nn
import numpy as np
import copy
import cv2
from .convnext import *
from .layers import ConvBlock3x3, Self_Attn, ConvBlock, upsample, upsample4
from collections import OrderedDict
import torch.nn.functional as F
import scipy.io as sio
def read_lut(kernel, h_size, index):
    if index ==1:
        LUT_name = 'G:\liujingguo\\five\LUT\LUT'+str(h_size) + '\LUT_'+str(kernel)+"x"+str(kernel)+'.mat'
        arr_LUT = sio.loadmat(LUT_name)['LUT']
    else:
        LUT_name = 'G:\liujingguo\\five\LUT\LUT' + str(h_size) + '\LUT_' + str(kernel) + "x" + str(kernel) + '_2.mat'
        arr_LUT = sio.loadmat(LUT_name)['LUT']
    return arr_LUT
def biliner(input=None,cooordinate=None):
    if input is None or cooordinate is None:return None
    B, C, height, width = input.shape
    out = []
    for i in range(B):
        new_in = input[i].permute(1,2,0)
        x = cooordinate[0]
        y = cooordinate[1]
        x1, y1 = torch.floor(x).type(torch.int), torch.floor(y).type(torch.int)
        x2, y2 = torch.as_tensor(x1+1,dtype=torch.int),torch.as_tensor(y1+1,dtype=torch.int)
        x1, x2 = torch.clamp(x1, 0, width - 1).type(torch.long),torch.clamp(x2, 0, width - 1).type(torch.long)
        y1, y2 = torch.clamp(y1, 0, height - 1).type(torch.long), torch.clamp(y2, 0, height - 1).type(torch.long)
        dx1, dx2 = x.type(torch.float) - x1, x2 - x.type(torch.float)
        dy1, dy2 = y.type(torch.float)- y1, y2 - y.type(torch.float)
        dx1, dx2 = dx1.unsqueeze(1).repeat(1,C),dx2.unsqueeze(1).repeat(1,C)
        dy1, dy2 = dy1.unsqueeze(1).repeat(1,C),dy2.unsqueeze(1).repeat(1,C)
        interpolated_value = (new_in[y1, x1] * dx2 * dy2 +
                              new_in[y1, x2] * dx1 * dy2 +
                              new_in[y2, x1] * dx2 * dy1 +
                              new_in[y2, x2] * dx1 * dy1)
        out.append(interpolated_value)
    out = torch.stack(out,dim=0)
    return out
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

class sinSeg(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self):
        super(sinSeg, self).__init__()

        self.num_ch_enc = np.array([128, 128, 256, 512, 1024])  #
        self.num_ch_dec = np.array([64, 128, 128, 256, 512])

        self.equi_dec_convs = OrderedDict()
        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0],
                                                    self.num_ch_dec[0])
        self.equi_dec_convs["segconv_0"] = nn.Conv2d(self.num_ch_dec[0], 13, 1)
        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
    def forward(self,equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4, msk = True):

        equi_x = upsample(self.equi_dec_convs["upconv_5"](equi_enc_feat4)) #16
        equi_x = torch.cat([equi_x, equi_enc_feat3], 1)

        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))
        equi_x = torch.cat([equi_x, equi_enc_feat2], 1)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))
        equi_x = torch.cat([equi_x, equi_enc_feat1], 1)
        equi_x = self.equi_dec_convs["upconv_2"](equi_x)
        equi_x = torch.cat([equi_x, equi_enc_feat0], 1)
        equi_x = upsample4(self.equi_dec_convs["upconv_1"](equi_x))
        out = self.equi_dec_convs["segconv_0"](equi_x)

        return out
class sinSeg_msk(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self):
        super(sinSeg_msk, self).__init__()

        self.num_ch_enc = np.array([96, 96, 192, 384, 768])  #
        self.num_ch_dec = np.array([48, 96, 96, 192, 384,])
        self.equi_dec_convs = OrderedDict()
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_enc[3], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_enc[2]+self.num_ch_dec[3], self.num_ch_dec[2])
        # self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[2]) ######两层使用
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_enc[1]+self.num_ch_dec[2], self.num_ch_dec[1])#
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[0])
        self.equi_dec_convs["segconv_0"] = nn.Conv2d(self.num_ch_dec[0], 1, 1)


        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
    ######################    original


    def forward(self,equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4):

        #
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_enc_feat3))
        equi_x = torch.cat([equi_x, equi_enc_feat2], 1)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))
        equi_x = torch.cat([equi_x, equi_enc_feat1], 1)

        equi_x = self.equi_dec_convs["upconv_2"](equi_x)
        equi_x = torch.cat([equi_x, equi_enc_feat0], 1)
        equi_x = upsample4(self.equi_dec_convs["upconv_1"](equi_x))
        out = self.equi_dec_convs["segconv_0"](equi_x)

        out = out.view(out.size(0), out.size(1), -1)
        out = F.softmax(out, dim=-1)
        out = out.view(out.size(0), out.size(1), 256,512)
        return out
class Equi_convnext_tea(nn.Module):
    def __init__(self, equi_h, equi_w, invalid_ids=[], num_classes=13, init_bias=0.0, **kwargs):
        super(Equi_convnext_tea, self).__init__()

        self.equi_encoder = convnext_base(pretrained=True)
        self.equi_encoder_sph = convnext_tiny(pretrained=True)
        # self.advpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(128, 10)
        self.equi_h = equi_h
        self.equi_w = equi_w
        # self.fusion_type = fusion_type
        # self.se_in_fusion = se_in_fusion
        self.invalid_ids = invalid_ids
        #########################
        # self.conv_next = nn.Conv2d(4, 128, kernel_size=4, stride=4)
        # self.conv_next2 = nn.Conv2d(4, 96, kernel_size=4, stride=4)
        # self.layernorm = LayerNorm(128, eps=1e-6,data_format="channels_first")
        # self.layernorm2 = LayerNorm(96, eps=1e-6,data_format="channels_first")
        # self.LUT4x4 = torch.from_numpy(np.array(read_lut(4, 256, 1))).cuda()
        # self.LUT4x4_2 = torch.from_numpy(np.array(read_lut(4, 256, 2))).cuda()
        # self.LUT2x2_64 = torch.from_numpy(np.array(read_lut(2, 64, 1))).cuda()
        # self.LUT2x2_2_64 = torch.from_numpy(np.array(read_lut(2, 64, 2))).cuda()
        # self.LUT2x2_32 = torch.from_numpy(np.array(read_lut(2, 32, 1))).cuda()
        # self.LUT2x2_2_32 = torch.from_numpy(np.array(read_lut(2, 32, 2))).cuda()
        # self.LUT2x2_16 = torch.from_numpy(np.array(read_lut(2, 16, 1))).cuda()
        # self.LUT2x2_2_16 = torch.from_numpy(np.array(read_lut(2, 16, 2))).cuda()
        ###########################################

        self.bias = nn.Parameter(torch.full([1, num_classes, 1, 1], init_bias))
        self.seg1 = sinSeg_msk()
        self.seg2 = sinSeg_msk()
        self.seg3 = sinSeg_msk()
        self.seg4 = sinSeg_msk()
        self.seg5 = sinSeg_msk()
        self.seg6 = sinSeg_msk()
        self.seg7 = sinSeg_msk()
        self.seg8 = sinSeg_msk()
        self.seg9 = sinSeg_msk()
        self.seg10 = sinSeg_msk()
        self.seg11 = sinSeg_msk()
        self.seg12 = sinSeg_msk()
        self.seg13 = sinSeg_msk()
        self.seg = sinSeg()


    def forward(self, input_equi_image):
        outputs = {}

        equi_enc_feat0,equi_enc_feat1,equi_enc_feat2,equi_enc_feat3,equi_enc_feat4,xout = self.equi_encoder(input_equi_image)
        equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph,xout_sph = self.equi_encoder_sph(input_equi_image)
        # equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3, equi_enc_feat4, xout = equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph,xout_sph
        ########################   RGB-D
        # B, C, H, W = input_equi_image.shape
        # x_left = biliner(input_equi_image, [self.LUT4x4[:, 0], self.LUT4x4[:, 1]])
        # x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x_right = biliner(input_equi_image, [self.LUT4x4_2[:, 0], self.LUT4x4_2[:, 1]])
        # x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x = torch.cat((x_left[:, :, :, :256], x_right[:, :, :, 256:]), dim=3)
        # equi_enc_feat0_sph = self.conv_next2(x)
        # equi_enc_feat0_sph = self.layernorm2(equi_enc_feat0_sph)
        # equi_enc_feat1_sph = self.equi_encoder_sph.stages[0](equi_enc_feat0_sph)
        # B, C, H, W = equi_enc_feat1_sph.shape
        # x_left = biliner(equi_enc_feat1_sph, [self.LUT2x2_64[:, 0], self.LUT2x2_64[:, 1]])
        # x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x_right = biliner(equi_enc_feat1_sph, [self.LUT2x2_2_64[:, 0], self.LUT2x2_2_64[:, 1]])
        # x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x = torch.cat((x_left[:, :, :, :64], x_right[:, :, :, 64:]), dim=3)
        # equi_enc_feat2_sph = self.equi_encoder_sph.stages[1](self.equi_encoder_sph.downsample_layers[1](x))
        # B, C, H, W = equi_enc_feat2_sph.shape
        # x_left = biliner(equi_enc_feat2_sph, [self.LUT2x2_32[:, 0], self.LUT2x2_32[:, 1]])
        # x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x_right = biliner(equi_enc_feat2_sph, [self.LUT2x2_2_32[:, 0], self.LUT2x2_2_32[:, 1]])
        # x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x = torch.cat((x_left[:, :, :, :32], x_right[:, :, :, 32:]), dim=3)
        # equi_enc_feat3_sph = self.equi_encoder_sph.stages[2](self.equi_encoder_sph.downsample_layers[2](x))
        # equi_enc_feat4_sph = equi_enc_feat3_sph
        # equi_enc_feat0 = self.conv_next(input_equi_image)
        # equi_enc_feat0 = self.layernorm(equi_enc_feat0)
        # equi_enc_feat1 = self.equi_encoder.stages[0](equi_enc_feat0)
        # equi_enc_feat2 = self.equi_encoder.stages[1](self.equi_encoder.downsample_layers[1](equi_enc_feat1))
        # equi_enc_feat3 = self.equi_encoder.stages[2](self.equi_encoder.downsample_layers[2](equi_enc_feat2))
        # equi_enc_feat4 = self.equi_encoder.stages[3](self.equi_encoder.downsample_layers[3](equi_enc_feat3))
        ############################################
        sem = self.seg(equi_enc_feat0,equi_enc_feat1,equi_enc_feat2,equi_enc_feat3,equi_enc_feat4)

        msk1 = self.seg1(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk2 = self.seg2(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk3 = self.seg3(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk4 = self.seg4(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk5 = self.seg5(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)

        msk6 = self.seg6(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk7 = self.seg7(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk8 = self.seg8(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk9 = self.seg9(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk10 = self.seg10(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk11 = self.seg11(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)

        msk12 = self.seg12(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)
        msk13 = self.seg13(equi_enc_feat0_sph,equi_enc_feat1_sph,equi_enc_feat2_sph,equi_enc_feat3_sph,equi_enc_feat4_sph)

        sem1 = F.relu(sem[:, 0, :, :] + msk1[:, 0, :, :] * sem[:, 0, :, :]).unsqueeze(1)
        sem2 = F.relu(sem[:, 1, :, :] + msk2[:, 0, :, :] * sem[:, 1, :, :]).unsqueeze(1)
        sem3 = F.relu(sem[:, 2, :, :] + msk3[:, 0, :, :] * sem[:, 2, :, :]).unsqueeze(1)
        sem4 = F.relu(sem[:, 3, :, :] + msk4[:, 0, :, :] * sem[:, 3, :, :]).unsqueeze(1)
        sem5 = F.relu(sem[:, 4, :, :] + msk5[:, 0, :, :] * sem[:, 4, :, :]).unsqueeze(1)
        sem6 = F.relu(sem[:, 5, :, :] + msk6[:, 0, :, :] * sem[:, 5, :, :]).unsqueeze(1)
        sem7 = F.relu(sem[:, 6, :, :] + msk7[:, 0, :, :] * sem[:, 6, :, :]).unsqueeze(1)
        sem8 = F.relu(sem[:, 7, :, :] + msk8[:, 0, :, :] * sem[:, 7, :, :]).unsqueeze(1)
        sem9 = F.relu(sem[:, 8, :, :] + msk9[:, 0, :, :] * sem[:, 8, :, :]).unsqueeze(1)
        sem10 = F.relu(sem[:, 9, :, :] + msk10[:, 0, :, :] * sem[:, 9, :, :]).unsqueeze(1)
        sem11 = F.relu(sem[:, 10, :, :] + msk11[:, 0, :, :] * sem[:, 10, :, :]).unsqueeze(1)
        sem12 = F.relu(sem[:, 11, :, :] + msk12[:, 0, :, :] * sem[:, 11, :, :]).unsqueeze(1)
        sem13 = F.relu(sem[:, 12, :, :] + msk13[:, 0, :, :] * sem[:, 12, :, :]).unsqueeze(1)


        sem = torch.cat([sem1, sem2, sem3, sem4, sem5, sem6, sem7, sem8, sem9, sem10, sem11, sem12, sem13], 1)
        #sem = self.final(sem)

        sem = self.bias + sem
        sem[:, self.invalid_ids] = -100
        outputs["sem"] = sem
        # out = self.fc(xout)
        outputs["equi_enc_feat0"] = equi_enc_feat0
        # outputs["result"] = out
        outputs["equi_enc_feat1"] = equi_enc_feat1
        outputs["equi_enc_feat2"] =equi_enc_feat2
        outputs["equi_enc_feat3"] = equi_enc_feat3
        outputs["equi_enc_feat4"] = equi_enc_feat4
        outputs["mask1"] = msk1
        outputs["mask2"] = msk2
        outputs["mask3"] = msk3
        outputs["mask4"] = msk4
        outputs["mask5"] = msk5
        outputs["mask6"] = msk6
        outputs["mask7"] = msk7
        outputs["mask8"] = msk8
        outputs["mask9"] = msk9
        outputs["mask10"] = msk10
        outputs["mask11"] = msk11
        outputs["mask12"] = msk12
        outputs["mask13"] = msk13
        return outputs

