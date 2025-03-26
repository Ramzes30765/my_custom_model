import torch.nn as nn
import torch.nn.functional as F

from modules.attention_modules import CBAM, AttentionAugmentedConv, SKConv


class DynamicHeadBlock(nn.Module):
    """
    Блок динамической головы, который состоит из динамической свёртки, 
    последующей адаптивной (канальной) коррекции через attention и остаточного соединения.
    """
    def __init__(self, channels, M=2, r=16):
        super().__init__()
        self.dynamic_conv = SKConv(channels, M, r)
        self.attention = CBAM(channels, M)

    def forward(self, x):
        conv_out = self.dynamic_conv(x)
        attn = self.attention(x)
        # Применяем attention, а затем добавляем исходный сигнал (остаточное соединение)
        out = conv_out * attn
        return out + x


class DynamicHead(nn.Module):
    """
    Динамическая голова детекции, которая состоит из стека DynamicHeadBlock.
    После этого блоки разделяются на две подсети для классификации и регрессии боксов.
    
    Аргументы:
      - in_channels: число входных каналов (например, 256)
      - num_blocks: число динамических блоков (например, 6)
      - num_classes: число классов детекции
      - num_anchors: число якорей (anchors) на каждую позицию
    """
    def __init__(self, in_channels, num_blocks=6, num_classes=80, num_anchors=9):
        super().__init__()
        self.blocks = nn.Sequential(*[DynamicHeadBlock(in_channels) for _ in range(num_blocks)])
        # Предсказание классов: выход = num_classes * num_anchors
        self.cls_pred = nn.Conv2d(in_channels, num_classes * num_anchors, kernel_size=3, padding=1)
        # Предсказание боксов: выход = 4 * num_anchors (x, y, w, h)
        self.box_pred = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=3, padding=1)

    def forward(self, x):
        # x — карта признаков от, например, BiFPN или FPN, размер (B, C, H, W)
        features = self.blocks(x)
        cls_out = self.cls_pred(features)
        box_out = self.box_pred(features)
        return cls_out, box_out