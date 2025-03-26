import timm
import torch
from torch import nn

class FPN(nn.Module):
    def __init__(self, backbone_name='resnet50', out_channels=256):
        super().__init__()
        # Создаем бэкбон с выводом C2-C5
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            out_indices=(1, 2, 3, 4),  # C2, C3, C4, C5
            pretrained=True
        )
        # Получаем количество каналов для каждого уровня
        self.channels = self.backbone.feature_info.channels()
        
        # Lateral 1x1 convolutions для выравнивания каналов
        self.lateral_c2 = nn.Conv2d(self.channels[0], out_channels, kernel_size=1)
        self.lateral_c3 = nn.Conv2d(self.channels[1], out_channels, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(self.channels[2], out_channels, kernel_size=1)
        self.lateral_c5 = nn.Conv2d(self.channels[3], out_channels, kernel_size=1)
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        
        # 3x3 convolutions для сглаживания
        self.smooth_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Получаем признаки C2-C5 из бэкбона
        c2, c3, c4, c5 = self.backbone(x)
        
        # Top-down pathway и lateral connections
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + self.up(p5)
        p4 = self.smooth_p4(p4)
        
        p3 = self.lateral_c3(c3) + self.up(p4)
        p3 = self.smooth_p3(p3)
        
        p2 = self.lateral_c2(c2) + self.up(p3)
        p2 = self.smooth_p2(p2)
        
        return p2, p3, p4, p5