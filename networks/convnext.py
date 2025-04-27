# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import numpy as np
import scipy.io as sio

def read_lut(kernel, h_size, index):
    if index ==1:
        LUT_name = '/gemini/code/F1/LUT/LUT'+str(h_size) + '/LUT_'+str(kernel)+"x"+str(kernel)+'.mat'
        arr_LUT = sio.loadmat(LUT_name)['LUT']
    else:
        LUT_name = '/gemini/code/F1/LUT/LUT' + str(h_size) + '/LUT_' + str(kernel) + "x" + str(kernel) + '_2.mat'
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
class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
class Block0(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.LUT7x7_64 = torch.from_numpy(np.array(read_lut(7,64,1))).cuda()
        self.LUT7x7_2_64 = torch.from_numpy(np.array(read_lut(7, 64, 2))).cuda()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=7, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        B, C, H, W = x.shape
        x_left = biliner(x, [self.LUT7x7_64[:, 0], self.LUT7x7_64[:, 1]])
        x_left = x_left.reshape(B, H * 7, W * 7, C).permute(0, 3, 1, 2)
        x_right = biliner(x, [self.LUT7x7_2_64[:, 0], self.LUT7x7_2_64[:, 1]])
        x_right = x_right.reshape(B, H * 7, W * 7, C).permute(0, 3, 1, 2)
        x = torch.cat((x_left[:, :, :, :(64 * 7)], x_right[:, :, :, (64 * 7):]), dim=3)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
class Block1(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.LUT7x7_32 = torch.from_numpy(np.array(read_lut(7,32,1))).cuda()
        self.LUT7x7_2_32 = torch.from_numpy(np.array(read_lut(7, 32, 2))).cuda()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=7, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        B, C, H, W = x.shape
        x_left = biliner(x, [self.LUT7x7_32[:, 0], self.LUT7x7_32[:, 1]])
        x_left = x_left.reshape(B, H * 7, W * 7, C).permute(0, 3, 1, 2)
        x_right = biliner(x, [self.LUT7x7_2_32[:, 0], self.LUT7x7_2_32[:, 1]])
        x_right = x_right.reshape(B, H * 7, W * 7, C).permute(0, 3, 1, 2)
        x = torch.cat((x_left[:, :, :, :(32 * 7)], x_right[:, :, :, (32 * 7):]), dim=3)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
class Block2(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.LUT7x7_16 = torch.from_numpy(np.array(read_lut(7,16,1))).cuda()
        self.LUT7x7_2_16 = torch.from_numpy(np.array(read_lut(7, 16, 2))).cuda()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=7, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        B, C, H, W = x.shape
        x_left = biliner(x, [self.LUT7x7_16[:, 0], self.LUT7x7_16[:, 1]])
        x_left = x_left.reshape(B, H * 7, W * 7, C).permute(0, 3, 1, 2)
        x_right = biliner(x, [self.LUT7x7_2_16[:, 0], self.LUT7x7_2_16[:, 1]])
        x_right = x_right.reshape(B, H * 7, W * 7, C).permute(0, 3, 1, 2)
        x = torch.cat((x_left[:, :, :, :(16 * 7)], x_right[:, :, :, (16 * 7):]), dim=3)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
class Block3(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.LUT7x7_8 = torch.from_numpy(np.array(read_lut(7,8,1))).cuda()
        self.LUT7x7_2_8 = torch.from_numpy(np.array(read_lut(7, 8, 2))).cuda()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, stride=7, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        B, C, H, W = x.shape
        x_left = biliner(x, [self.LUT7x7_8[:, 0], self.LUT7x7_8[:, 1]])
        x_left = x_left.reshape(B, H * 7, W * 7, C).permute(0, 3, 1, 2)
        x_right = biliner(x, [self.LUT7x7_2_8[:, 0], self.LUT7x7_2_8[:, 1]])
        x_right = x_right.reshape(B, H * 7, W * 7, C).permute(0, 3, 1, 2)
        x = torch.cat((x_left[:, :, :, :(8 * 7)], x_right[:, :, :, (8 * 7):]), dim=3)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            if i == 0:
                x0 = x
            x = self.stages[i](x)
            if i == 0:
                x1 = x
                # x5 = x1 ####################
            elif i == 1:
                x2 = x
            elif i == 2:
                x3 = x
            else:
                x4 = x
        xout = x4
        return x0,x1,x2,x3,x4,xout

    def forward(self, x):
        x0,x1,x2,x3,x4,xout = self.forward_features(x)
        # x = self.head(x)
        # x = self.does[0](x)
        # x = self.stage[0](x)
        return x0,x1,x2,x3,x4,xout

class ConvNeXt_sph(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            if i==0:
                stage = nn.Sequential(
                    *[Block0(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i==1:
                stage = nn.Sequential(
                    *[Block1(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i==2:
                stage = nn.Sequential(
                    *[Block2(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i==3:
                stage = nn.Sequential(
                    *[Block3(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                        *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.LUT4x4 = torch.from_numpy(np.array(read_lut(4,256,1))).cuda()
        self.LUT4x4_2 = torch.from_numpy(np.array(read_lut(4,256,2))).cuda()
        self.LUT2x2_64 = torch.from_numpy(np.array(read_lut(2, 64, 1))).cuda()
        self.LUT2x2_2_64 = torch.from_numpy(np.array(read_lut(2, 64, 2))).cuda()
        self.LUT2x2_32 = torch.from_numpy(np.array(read_lut(2, 32,1))).cuda()
        self.LUT2x2_2_32 = torch.from_numpy(np.array(read_lut(2, 32,2))).cuda()
        self.LUT2x2_16 = torch.from_numpy(np.array(read_lut(2, 16,1))).cuda()
        self.LUT2x2_2_16 = torch.from_numpy(np.array(read_lut(2, 16,2))).cuda()
        # self.LUT2x2_8 = read_lut(2, 8, 1)
        # self.LUT2x2_2_8 = read_lut(2, 8, 1)
        # kernel = [[1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480],
        #           [1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240],
        #           [1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120],
        #           [1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60],
        #           [1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60],
        #           [1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120],
        #           [1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240],
        #           [1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480]
        # ]
        # self.kernel_weight = torch.FloatTensor(kernel).expand(1024,1024,8,16).cuda()
        # self.kernel_weight = nn.Parameter(data = kernel, requires_grad=False)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(3):####### original foe i in range 3
            if i == 0:
                B,C,H,W=x.shape
                # coordinate = [self.LUT4x4[:,0], self.LUT4x4[:,1]]
                x_left = biliner(x,[self.LUT4x4[:,0], self.LUT4x4[:,1]])
                x_left = x_left.reshape(B,H,W,C).permute(0,3,1,2)
                x_right = biliner(x, [self.LUT4x4_2[:, 0], self.LUT4x4_2[:, 1]])
                x_right = x_right.reshape(B,H,W,C).permute(0,3,1,2)
                x = torch.cat((x_left[:,:,:,:512],x_right[:,:,:,512:]),dim=3)
                # cv2.imwrite(,x.cpu().squeeze(0).permute(1,2,0).numpy())
                x = self.downsample_layers[i](x)
            elif i==1:
                B, C, H, W = x.shape
                x_left = biliner(x, [self.LUT2x2_64[:, 0], self.LUT2x2_64[:, 1]])
                x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x_right = biliner(x, [self.LUT2x2_2_64[:, 0], self.LUT2x2_2_64[:, 1]])
                x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = torch.cat((x_left[:, :, :, :256], x_right[:, :, :, 256:]), dim=3)
                x = self.downsample_layers[i](x)
            elif i==2:
                B, C, H, W = x.shape
                x_left = biliner(x, [self.LUT2x2_32[:, 0], self.LUT2x2_32[:, 1]])
                x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x_right = biliner(x, [self.LUT2x2_2_32[:, 0], self.LUT2x2_2_32[:, 1]])
                x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = torch.cat((x_left[:, :, :, :32], x_right[:, :, :, 32:]), dim=3)
                x = self.downsample_layers[i](x)
            elif i==3:
                B, C, H, W = x.shape
                x_left = biliner(x, [self.LUT2x2_16[:, 0], self.LUT2x2_16[:, 1]])
                x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x_right = biliner(x, [self.LUT2x2_2_16[:, 0], self.LUT2x2_2_16[:, 1]])
                x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = torch.cat((x_left[:, :, :, :16], x_right[:, :, :, 16:]), dim=3)
                x = self.downsample_layers[i](x)
            else:
                x = self.downsample_layers[i](x)
            if i == 0:
                x0 = x
                x = self.stages[i](x)
                x1 = x
                # x2, x3, x4 = x1, x1, x1
                # x5 = x1 ####################
            #########   experiments
            if i == 1: #########
                x = self.stages[i](x)
                x2 = x
                # x3 = x
                # x4 = x
            if i == 2:
                x = self.stages[i](x)
                x3= x
                x4 = x
            # if i==3:
            #     x = self.stages[i](x)
            #     x4 = x
            # elif i == 2:
            #     x4 = x

            # x4 = x

        # return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

        # xout = self.norm(F.conv2d(x4, self.kernel_weight).view(1, 768))
        # xout = self.norm(x4.mean([-2, -1])) #######meiyong
        xout = x4
        return x0,x1,x2,x3,x4,xout

    def forward(self, x):
        x0,x1,x2,x3,x4,xout = self.forward_features(x)
        # x = self.head(x)
        # x = self.does[0](x)
        # x = self.stage[0](x)
        return x0,x1,x2,x3,x4,xout
class ConvNeXt_sph1(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            if i==0:
                stage = nn.Sequential(
                    *[Block0(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i==1:
                stage = nn.Sequential(
                    *[Block1(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i==2:
                stage = nn.Sequential(
                    *[Block2(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            # elif i==3:
            #     stage = nn.Sequential(
            #         *[Block3(dim=dims[i], drop_path=dp_rates[cur + j],
            #                 layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            #     )
            else:
                stage = nn.Sequential(
                        *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.LUT4x4 = torch.from_numpy(np.array(read_lut(4,256,1))).cuda()
        self.LUT4x4_2 = torch.from_numpy(np.array(read_lut(4,256,2))).cuda()
        self.LUT2x2_64 = torch.from_numpy(np.array(read_lut(2, 64, 1))).cuda()
        self.LUT2x2_2_64 = torch.from_numpy(np.array(read_lut(2, 64, 2))).cuda()
        self.LUT2x2_32 = torch.from_numpy(np.array(read_lut(2, 32,1))).cuda()
        self.LUT2x2_2_32 = torch.from_numpy(np.array(read_lut(2, 32,2))).cuda()
        self.LUT2x2_16 = torch.from_numpy(np.array(read_lut(2, 16,1))).cuda()
        self.LUT2x2_2_16 = torch.from_numpy(np.array(read_lut(2, 16,2))).cuda()
        # self.LUT2x2_8 = read_lut(2, 8, 1)
        # self.LUT2x2_2_8 = read_lut(2, 8, 1)
        # kernel = [[1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480],
        #           [1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240],
        #           [1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120],
        #           [1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60],
        #           [1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60],
        #           [1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120],
        #           [1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240],
        #           [1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480]
        # ]
        # self.kernel_weight = torch.FloatTensor(kernel).expand(1024,1024,8,16).cuda()
        # self.kernel_weight = nn.Parameter(data = kernel, requires_grad=False)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(1):####### original foe i in range 1
            if i == 0:
                B,C,H,W=x.shape
                # coordinate = [self.LUT4x4[:,0], self.LUT4x4[:,1]]
                x_left = biliner(x,[self.LUT4x4[:,0], self.LUT4x4[:,1]])
                x_left = x_left.reshape(B,H,W,C).permute(0,3,1,2)
                x_right = biliner(x, [self.LUT4x4_2[:, 0], self.LUT4x4_2[:, 1]])
                x_right = x_right.reshape(B,H,W,C).permute(0,3,1,2)
                x = torch.cat((x_left[:,:,:,:256],x_right[:,:,:,256:]),dim=3)
                # cv2.imwrite(,x.cpu().squeeze(0).permute(1,2,0).numpy())
                x = self.downsample_layers[i](x)
            # elif i==1:
            #     B, C, H, W = x.shape
            #     x_left = biliner(x, [self.LUT2x2_64[:, 0], self.LUT2x2_64[:, 1]])
            #     x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
            #     x_right = biliner(x, [self.LUT2x2_2_64[:, 0], self.LUT2x2_2_64[:, 1]])
            #     x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
            #     x = torch.cat((x_left[:, :, :, :64], x_right[:, :, :, 64:]), dim=3)
            #     x = self.downsample_layers[i](x)
            # elif i==2:
            #     B, C, H, W = x.shape
            #     x_left = biliner(x, [self.LUT2x2_32[:, 0], self.LUT2x2_32[:, 1]])
            #     x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
            #     x_right = biliner(x, [self.LUT2x2_2_32[:, 0], self.LUT2x2_2_32[:, 1]])
            #     x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
            #     x = torch.cat((x_left[:, :, :, :32], x_right[:, :, :, 32:]), dim=3)
            #     x = self.downsample_layers[i](x)
            # elif i==3:
            #     B, C, H, W = x.shape
            #     x_left = biliner(x, [self.LUT2x2_16[:, 0], self.LUT2x2_16[:, 1]])
            #     x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
            #     x_right = biliner(x, [self.LUT2x2_2_16[:, 0], self.LUT2x2_2_16[:, 1]])
            #     x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
            #     x = torch.cat((x_left[:, :, :, :16], x_right[:, :, :, 16:]), dim=3)
            #     x = self.downsample_layers[i](x)
            else:
                x = self.downsample_layers[i](x)
            if i == 0:
                x0 = x
                x = self.stages[i](x)
                x1 = x
                x2, x3, x4 = x1,x1,x1
                # x5 = x1 ####################
            #########   experiments
            # if i == 1: #########
            #     x = self.stages[i](x)
            #     x2 = x
            #     # x3 = x
            #     # x4 = x
            # if i==2:
            #     x = self.stages[i](x)
            #     x3= x
            #     # x4 = x
            # if i==3:
            #     x = self.stages[i](x)
            #     x4 = x
            # elif i == 2:
            #     x4 = x
            # if i==4:
            #     x = self.stages[i](x)
            # x4 = x

        # return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

        # xout = self.norm(F.conv2d(x4, self.kernel_weight).view(1, 768))
        # xout = self.norm(x4.mean([-2, -1])) #######meiyong
        xout = x4
        return x0,x1,x2,x3,x4,xout

    def forward(self, x):
        x0,x1,x2,x3,x4,xout = self.forward_features(x)
        # x = self.head(x)
        # x = self.does[0](x)
        # x = self.stage[0](x)
        return x0,x1,x2,x3,x4,xout

class ConvNeXt_sph_omni(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            if i==0:
                stage = nn.Sequential(
                    *[Block0(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i==1:
                stage = nn.Sequential(
                    *[Block1(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i==2:
                stage = nn.Sequential(
                    *[Block2(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            elif i==3:
                stage = nn.Sequential(
                    *[Block3(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                        *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        self.LUT4x4 = torch.from_numpy(np.array(read_lut(4,256,1))).cuda()
        self.LUT4x4_2 = torch.from_numpy(np.array(read_lut(4,256,2))).cuda()
        self.LUT2x2_64 = torch.from_numpy(np.array(read_lut(2, 64, 1))).cuda()
        self.LUT2x2_2_64 = torch.from_numpy(np.array(read_lut(2, 64, 2))).cuda()
        self.LUT2x2_32 = torch.from_numpy(np.array(read_lut(2, 32,1))).cuda()
        self.LUT2x2_2_32 = torch.from_numpy(np.array(read_lut(2, 32,2))).cuda()
        self.LUT2x2_16 = torch.from_numpy(np.array(read_lut(2, 16,1))).cuda()
        self.LUT2x2_2_16 = torch.from_numpy(np.array(read_lut(2, 16,2))).cuda()
        # self.LUT2x2_8 = read_lut(2, 8, 1)
        # self.LUT2x2_2_8 = read_lut(2, 8, 1)
        # kernel = [[1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480],
        #           [1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240],
        #           [1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120],
        #           [1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60],
        #           [1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60,  1./ 60],
        #           [1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120, 1./ 120],
        #           [1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240, 1./ 240],
        #           [1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480, 1./ 480]
        # ]
        # self.kernel_weight = torch.FloatTensor(kernel).expand(1024,1024,8,16).cuda()
        # self.kernel_weight = nn.Parameter(data = kernel, requires_grad=False)
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):####### original foe i in range 1
            if i == 0:
                B,C,H,W=x.shape
                # coordinate = [self.LUT4x4[:,0], self.LUT4x4[:,1]]
                x_left = biliner(x,[self.LUT4x4[:,0], self.LUT4x4[:,1]])
                x_left = x_left.reshape(B,H,W,C).permute(0,3,1,2)
                x_right = biliner(x, [self.LUT4x4_2[:, 0], self.LUT4x4_2[:, 1]])
                x_right = x_right.reshape(B,H,W,C).permute(0,3,1,2)
                x = torch.cat((x_left[:,:,:,:256],x_right[:,:,:,256:]),dim=3)
                # cv2.imwrite(,x.cpu().squeeze(0).permute(1,2,0).numpy())
                x = self.downsample_layers[i](x)
            elif i==1:
                B, C, H, W = x.shape
                x_left = biliner(x, [self.LUT2x2_64[:, 0], self.LUT2x2_64[:, 1]])
                x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x_right = biliner(x, [self.LUT2x2_2_64[:, 0], self.LUT2x2_2_64[:, 1]])
                x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = torch.cat((x_left[:, :, :, :64], x_right[:, :, :, 64:]), dim=3)
                x = self.downsample_layers[i](x)
            elif i==2:
                B, C, H, W = x.shape
                x_left = biliner(x, [self.LUT2x2_32[:, 0], self.LUT2x2_32[:, 1]])
                x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x_right = biliner(x, [self.LUT2x2_2_32[:, 0], self.LUT2x2_2_32[:, 1]])
                x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = torch.cat((x_left[:, :, :, :32], x_right[:, :, :, 32:]), dim=3)
                x = self.downsample_layers[i](x)
            elif i==3:
                B, C, H, W = x.shape
                x_left = biliner(x, [self.LUT2x2_16[:, 0], self.LUT2x2_16[:, 1]])
                x_left = x_left.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x_right = biliner(x, [self.LUT2x2_2_16[:, 0], self.LUT2x2_2_16[:, 1]])
                x_right = x_right.reshape(B, H, W, C).permute(0, 3, 1, 2)
                x = torch.cat((x_left[:, :, :, :16], x_right[:, :, :, 16:]), dim=3)
                x = self.downsample_layers[i](x)
            else:
                x = self.downsample_layers[i](x)
            if i == 0:
                x0 = x
                x = self.stages[i](x)
                x1 = x
                # x5 = x1 ####################
            #########   experiments
            if i == 1: #########
                x = self.stages[i](x)
                x2 = x
                # x3 = x
                # x4 = x
            if i==2:
                x = self.stages[i](x)
                x3= x
            #     x4 = x
            if i==3:
                x = self.stages[i](x)
                x4 = x
            # elif i == 2:
            #     x4 = x
            # if i==4:
            #     x = self.stages[i](x)
            # x4 = x

        # return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

        # xout = self.norm(F.conv2d(x4, self.kernel_weight).view(1, 768))
        xout = self.norm(x4.mean([-2, -1])) #######meiyong
        # xout = x4
        return x0,x1,x2,x3,x4,xout

    def forward(self, x):
        x0,x1,x2,x3,x4,xout = self.forward_features(x)
        # x = self.head(x)
        # x = self.does[0](x)
        # x = self.stage[0](x)
        return x0,x1,x2,x3,x4,xout

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


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}


@register_model
def convnext_tiny(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_sph( depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model
@register_model
def convnext_tiny_omni(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt_sph_omni( depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def convnext_small(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        print("use pretrained model")
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model