import torch
import torch.nn as nn
from modules.attention_modules import SKConv


class ScaleAwareAttention(nn.Module):
    def __init__(self, in_channels, num_levels):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels * num_levels, num_levels, kernel_size=1)
        self.sigmoid = nn.Hardsigmoid()

    def forward(self, feats):  # list of [B, C, H, W]
        pooled = [self.global_pool(f) for f in feats]
        concat = torch.cat(pooled, dim=1)  # (B, C*num_levels, 1, 1)
        weights = self.sigmoid(self.conv(concat))  # (B, num_levels, 1, 1)
        weights = weights.split(1, dim=1)
        out = [w * f for w, f in zip(weights, feats)]
        return out


class SpatialAwareAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.skconv = SKConv(in_channels)

    def forward(self, x):
        feat = self.skconv(x)
        attn = torch.sigmoid(feat)
        return x * attn


class TaskAwareAttention(nn.Module):
    def __init__(self, in_channels, num_tasks=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, in_channels * num_tasks)
        )
        self.num_tasks = num_tasks
        self.in_channels = in_channels

    def forward(self, x):  # x: [B, C, H, W]
        B, C, _, _ = x.shape
        pooled = self.pool(x).view(B, C)
        task_weights = self.fc(pooled).view(B, self.num_tasks, C, 1, 1)
        outputs = [x * task_weights[:, i] for i in range(self.num_tasks)]
        return outputs
    

class DynamicHeadBlock(nn.Module):
    def __init__(self, in_channels, num_levels, num_tasks=2):
        super().__init__()
        self.scale_attn = ScaleAwareAttention(in_channels, num_levels)
        self.spatial_attn = nn.ModuleList([
            SpatialAwareAttention(in_channels) for _ in range(num_levels)
        ])
        self.task_attn = nn.ModuleList([
            TaskAwareAttention(in_channels, num_tasks) for _ in range(num_levels)
        ])

    def forward(self, feats):  # list of [B, C, H, W]
        feats = self.scale_attn(feats)
        feats = [sa(f) for sa, f in zip(self.spatial_attn, feats)]
        task_feats = [ta(f) for ta, f in zip(self.task_attn, feats)]
        cls_feats = [t[0] for t in task_feats]
        reg_feats = [t[1] for t in task_feats]
        return cls_feats, reg_feats


class DynamicHead(nn.Module):
    def __init__(self, in_channels, num_levels, num_blocks=6, num_tasks=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            DynamicHeadBlock(in_channels, num_levels, num_tasks)
            for _ in range(num_blocks)
        ])

    def forward(self, feats):  # feats = [P3, P4, ..., P7]
        for block in self.blocks:
            cls_feats, reg_feats = block(feats)
            # Смешиваем признаки для следующего блока
            feats = [(c + r) / 2 for c, r in zip(cls_feats, reg_feats)]
        return cls_feats, reg_feats