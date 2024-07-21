# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from fairscale.nn.checkpoint import checkpoint_wrapper
from ..builder import NECKS


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


@NECKS.register_module()
class SFP(BaseModule):
    r"""
    Our custom implementation of Simple Feature Pyramid (SFP).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 use_p2=False,
                 use_act_checkpoint=False,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SFP, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.use_p2 = use_p2
        self.fp16_enabled = False
        if self.use_p2:
            self.p2 = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.in_channels[0], self.in_channels[0]//2, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.in_channels[0]//2),
                nn.GELU(),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(self.in_channels[0]//2, self.in_channels[0]//4, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.in_channels[0]//4),
                nn.GELU(),
                nn.Conv2d(self.in_channels[0]//4, self.out_channels, kernel_size=1, bias=False),
                LayerNorm(self.out_channels),
                nn.GELU(),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
                LayerNorm(self.out_channels)
            )

        self.p3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.in_channels[0], self.in_channels[0]//2, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.in_channels[0]//2),
            nn.GELU(),
            nn.Conv2d(self.in_channels[0]//2, self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        self.p4 = nn.Sequential(
            nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        self.p5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        self.p6 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.in_channels[0], self.in_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            LayerNorm(self.in_channels[0]),
            nn.GELU(),
            nn.Conv2d(self.in_channels[0], self.out_channels, kernel_size=1, bias=False),
            LayerNorm(self.out_channels),
            nn.GELU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, bias=False),
            LayerNorm(self.out_channels)
        )
        
        if use_act_checkpoint:
            self.p3 = checkpoint_wrapper(self.p3)
            self.p4 = checkpoint_wrapper(self.p4)
            self.p5 = checkpoint_wrapper(self.p5)
            self.p6 = checkpoint_wrapper(self.p6)
            if self.use_p2:
                self.p2 = checkpoint_wrapper(self.p2)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        x = inputs[0]
        p4 = self.p4(x)
        p3 = self.p3(x)
        p5 = self.p5(x)
        p6 = self.p6(x)
        outs = [p3, p4, p5, p6]
        if self.use_p2:
            outs = [self.p2(x)] + outs
        return tuple(outs)


