import torch
from torch import nn
from torch.nn import functional as F
import timm

from modules.attention_modules import CBAM, AttentionAugmentedConv, SKConv
from necks.attention_bifpn import BiFPN
from heads.dynamic_head import DynamicHead

class SOTAModel(nn.Module):
    def __init__(self):
        self.neck = BiFPN(
            backbone='resnet50',
            out_channels=256,
            num_layers=1,
            use_lateral_cbam=True,
            use_td_cbam=True,
            use_skconv=True,
            use_aac=True
            )
        self.head = DynamicHead(
            in_channels=256,
            num_blocks=6,
            num_classes=80,
            num_anchors=9
            )