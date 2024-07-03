# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS
import torch.nn.functional as F
import time
import pdb
from .attention_3d import *

class ASPP_3D(nn.Module):
    def __init__(self, in_channel=32, depth=16):
        super(ASPP_3D, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv3d(in_channel, depth, 3, 1, padding=1, dilation=1)
        self.atrous_block3 = nn.Conv3d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block6 = nn.Conv3d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv3d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv3d(in_channel, depth, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv3d(depth * 5, in_channel, 1, 1)
    def forward(self, x):
        atrous_block1 = self.atrous_block1(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        net = self.conv_1x1_output(torch.cat([ atrous_block1,  atrous_block3, atrous_block6,
                                            atrous_block12, atrous_block18   ], dim=1))
        return net


@NECKS.register_module()
class SECONDFPN3D(BaseModule):
    """FPN used in SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (list[int]): Input channels of multi-scale feature maps.
        out_channels (list[int]): Output channels of feature maps.
        upsample_strides (list[int]): Strides used to upsample the
            feature maps.
        norm_cfg (dict): Config dict of normalization layers.
        upsample_cfg (dict): Config dict of upsample layers.
        conv_cfg (dict): Config dict of conv layers.
        use_conv_for_no_stride (bool): Whether to use conv when stride is 1.
    """
    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 upsample_cfg=dict(type='deconv3d', bias=False),
                 conv_cfg=dict(type='Conv3d', bias=False),
                 use_conv_for_no_stride=False,
                 use_output_upsample=False,
                 with_cp=False,
                 init_cfg=None):
        
        # replacing GN with BN3D, performance drops from 42.5 to 40.9. 
        # the difference may be exaggerated because the performance can fluncate a lot
        
        super(SECONDFPN3D, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fp16_enabled = False
        self.with_cp = with_cp

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=upsample_strides[i],
                    stride=upsample_strides[i])
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(
                upsample_layer, build_norm_layer(norm_cfg, out_channel)[1], nn.ReLU(inplace=True))
            
            deblocks.append(deblock)
        
        self.deblocks = nn.ModuleList(deblocks)
        
        self.use_output_upsample = use_output_upsample
        if self.use_output_upsample:
            output_channel = sum(out_channels)
            self.output_deblock = nn.Sequential(
                build_upsample_layer(
                    upsample_cfg, in_channels=output_channel,
                    out_channels=output_channel, kernel_size=2, stride=2),
                build_norm_layer(norm_cfg, output_channel)[1],
                nn.ReLU(inplace=True),
                # build_conv_layer(conv_cfg, in_channels=output_channel,
                #             out_channels=output_channel, kernel_size=3, stride=1, padding=1),
                # build_norm_layer(norm_cfg, output_channel)[1],
                # nn.ReLU(inplace=True),
            )

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]
        # self.aspp3d = ASPP_3D(in_channel=384, depth=128 )

        self.alpha = nn.Parameter( torch.zeros(1) )
        self.attention_3d = LinearAttention3D( query_dim=384, dim=384,  heads=2 )



    @auto_fp16()
    def forward(self, x, depth, temporal_voxel=None):
        """Forward function.

        Args:
            x (torch.Tensor): 4D Tensor in (N, C, H, W) shape.

        Returns:
            list[torch.Tensor]: Multi-level feature maps.
        """
        assert len(x) == len(self.in_channels)
        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]
        
        if len(ups) > 1:
            out = torch.cat(ups, dim=1) ## [4, 128, 128, 128, 16] * 3
        else:
            out = ups[0]
        
        if self.use_output_upsample: ## False
            out = torch.utils.checkpoint.checkpoint(self.output_deblock, out)

        out = self.alpha * self.attention_3d( query = out,  x = temporal_voxel[0] ) + out

        # out = temporal_voxel[0] + out  #### fuse temporal

        # out = self.fuse_3d( torch.cat((temporal_voxel[0], out),dim=1) ) #### fuse temporal

        
        return [out]  ## [4, 384, 128, 128, 16]   B C D H W
