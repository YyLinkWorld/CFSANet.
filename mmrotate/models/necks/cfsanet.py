# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch

from mmrotate.models.builder import ROTATED_NECKS

class SFEM(nn.Module):
    def __init__(self, in_channels):
        super(SFEM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels//2, 3, padding=1)

        self.conv_w = nn.Conv2d(2,2,3,padding=1)
        self.conv_last =nn.Conv2d(in_channels//2, in_channels, 1)
        self.alpha = nn.parameter.Parameter(torch.tensor(0.5))
    
    def forward(self, x_high, x_low):
        
        attn1 = self.conv1(x_high)
        attn2 = self.conv2(x_low)

        x_mix = torch.cat([attn1,attn2], dim=1)

        x_max = torch.max(x_mix, dim=1, keepdim=True)[0]
        x_avg = torch.mean(x_mix, dim=1, keepdim=True)

        x_w = torch.cat([x_max, x_avg], dim=1)
        std = torch.std(x_w, dim=1, keepdim=True)
        x_new = x_w.clone()
        x_new[:,0,:,:] = x_w[:,0,:,:] + std * self.alpha
        x_w = self.conv_w(x_new).sigmoid()
        w = attn1 * x_w[:,0,:,:].unsqueeze(1) + attn2 * x_w[:,1,:,:].unsqueeze(1)
        out = self.conv_last(w)

        return out


class CFEM(nn.Module):
    def __init__(self, in_channels):
        super(CFEM, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(),
            nn.Linear(in_channels//2, in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x_high, x_low):

        x_mix = x_high + x_low  

        b,c,_,_ = x_mix.size()

        y = self.avgpool(x_mix).view(b,c)
        y = self.mlp(y).view(b,c,1,1)

        return x_low * y.expand_as(x_low)

class ISCEM(nn.Module):#传入低噪声和高噪声的通道数，首先将高噪声上采样到低噪声的大小，然后分别进行空间注意力和通道注意力，进行特征混合
    def __init__(self, high_noise_in_channel, low_noise_in_channel, scale_factor=2):
        super(ISCEM, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        self.conv1 = nn.Conv2d(high_noise_in_channel, low_noise_in_channel, 1)
        self.spatial = SFEM(low_noise_in_channel)
        self.channel = CFEM(low_noise_in_channel)

        self.conv = nn.Conv2d(low_noise_in_channel * 2, high_noise_in_channel, 3, stride=2, padding=1)
    
    def Upsample(self, x):
        x = self.conv1(self.upsample(x))
        return x
    
    def forward(self, x_high, x_low):
        x_high_t = self.Upsample(x_high)
        x_spatial = self.spatial(x_high_t, x_low)
        x_chan = self.channel(x_high_t, x_low)
        x_mix = torch.cat([x_spatial, x_chan], dim=1)
        x_mix = self.conv(x_mix)
        out = x_mix + x_high

        return out
    
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class MSTFF3(nn.Module):
    def __init__(self,in_channels):
        super(MSTFF3, self).__init__()
        self.conv = BasicConv(in_channels, in_channels//2, 1, padding=0)
        self.evalue = nn.Conv2d(in_channels//2 * 3, 3, 1, 1)

    def forward(self, x1, x2, x3, x4):
        attn2 = self.conv(x2)
        attn3 = self.conv(x3)
        attn4 = self.conv(x4)

        x_mix = torch.cat([attn2,attn3,attn4], dim=1)
        weight = self.evalue(x_mix).softmax(dim=1)

        out = x2 * weight[:,0,:,:].unsqueeze(1) + x3 * weight[:,1,:,:].unsqueeze(1) + x4 * weight[:,2,:,:].unsqueeze(1) + x1

        return out
    
class MSTFF2(nn.Module):
    def __init__(self, in_channels):
        super(MSTFF2, self).__init__()
        self.conv = BasicConv(in_channels, in_channels//2, 1, padding=0)
        self.evalue = nn.Conv2d(in_channels//2 * 2, 3, 1, 1)

    def forward(self,x2, x3, x4):
        attn3 = self.conv(x3)
        attn4 = self.conv(x4)

        x_mix = torch.cat([attn3,attn4], dim=1)
        weight = self.evalue(x_mix).softmax(dim=1)

        out = x3 * weight[:,1,:,:].unsqueeze(1) + x4 * weight[:,2,:,:].unsqueeze(1) + x2

        return out
    
class MSTFF1(nn.Module):
    def __init__(self, in_channels):
        super(MSTFF1, self).__init__()
        self.conv = BasicConv(in_channels, in_channels//2, 1, padding=0)
        self.evalue = nn.Conv2d(in_channels//2 * 1, 3, 1, 1)

    def forward(self, x3, x4):
        attn4 = self.conv(x4)

        weight = self.evalue(attn4).softmax(dim=1)

        out = x4 * weight[:,2,:,:].unsqueeze(1) + x3

        return out

# class Upsample_nx(nn.Module):
#     def __init__(self, in_channels, out_channels, scale_factor=2):
#         super(Upsample_nx, self).__init__()
#         self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
#         self.conv = nn.Conv2d(in_channels, out_channels, 1)
    
#     def forward(self, x):
#         x = self.conv(self.upsample(x))
#         return x



@ROTATED_NECKS.register_module()
class CFSANet(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(CFSANet, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.normal_fpn = FPN(in_channels, out_channels, num_outs, start_level, end_level, add_extra_convs, relu_before_extra_convs,\
                               no_norm_on_lateral, conv_cfg, norm_cfg, act_cfg, upsample_cfg, init_cfg)
        
        self.iscem2 = ISCEM(in_channels[1], in_channels[0])
        self.iscem3 = ISCEM(in_channels[2], in_channels[1])
        self.iscem4 = ISCEM(in_channels[3], in_channels[2])


        self.mstff3 = MSTFF3(in_channels[0])
        self.mstff2 = MSTFF2(in_channels[1])
        self.mstff1 = MSTFF1(in_channels[2])


    def Upsample_nx(self, x, nx, in_channels, out_channels):
        x = nn.Upsample(scale_factor=nx, mode='bilinear')(x)
        conv = nn.Conv2d(in_channels, out_channels, 1)
        conv = conv.to(x.device)  # 确保conv层与x在同一设备上
        x = conv(x)
        
        return x

    def forward(self, inputs):
        x0,x1,x2,x3 = inputs
        outs = [inputs[0]]

        outs.append(self.iscem2(inputs[1], inputs[0]))
        outs.append(self.iscem3(inputs[2], inputs[1]))
        outs.append(self.iscem4(inputs[3], inputs[2]))

        x3_up2 = self.Upsample_nx(x3, 2, x3.size(1), x2.size(1))
        x3_up4 = self.Upsample_nx(x3, 4, x3.size(1), x1.size(1))
        x3_up8 = self.Upsample_nx(x3, 8, x3.size(1), x0.size(1))

        x2_up2 = self.Upsample_nx(x2, 2, x2.size(1), x1.size(1))
        x2_up4 = self.Upsample_nx(x2, 4, x2.size(1), x0.size(1))

        x1_up2 = self.Upsample_nx(x1, 2, x1.size(1), x0.size(1))

        outs[0] = self.mstff3(outs[0], x1_up2, x2_up4, x3_up8)
        outs[1] = self.mstff2(outs[1], x2_up2, x3_up4)
        outs[2] = self.mstff1(outs[2], x3_up2)
        
        return self.normal_fpn(tuple(outs))
                    


# @NECKS.register_module()
class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


# import torch

# if __name__ == '__main__':
#     x1=torch.randn(1,64,256,256)
#     x2=torch.randn(1,128,128,128)
#     x3=torch.randn(1,320,64,64)
#     x4=torch.randn(1,512,32,32)

#     inputs=tuple([x1,x2,x3,x4])
#     model = SETPyNet([64,128,320,512],256,5)

#     outputs=model(inputs)
#     for i in range(len(outputs)):
#         print(f'outputs[{i}].shape = {outputs[i].shape}')