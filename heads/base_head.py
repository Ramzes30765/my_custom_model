import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.attention_modules import CBAM


class DetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, conv_channels=256):
        """
        in_channels: число входных каналов (полученных от neck)
        num_classes: число классов объектов
        conv_channels: число фильтров в промежуточных слоях
        """
        super(DetectionHead, self).__init__()
        
        # Общая сверточная база для обеих ветвей
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True)
        )
        
        # Модуль внимания (можно заменить более сложными, например, self-attention)
        self.attention = CBAM(conv_channels)
        
        # Ветвь для классификации: предсказывает logits для каждого класса
        self.cls_branch = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, num_classes, kernel_size=1)
        )
        
        # Ветвь для регрессии: предсказывает 4 координаты бокса и objectness score (всего 5 каналов)
        self.reg_branch = nn.Sequential(
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_channels, 5, kernel_size=1)
        )
        
    def forward(self, x):
        """
        x: входной feature map размера [B, in_channels, H, W]
        Возвращает:
            cls_out: [B, num_classes, H, W] – логиты классов
            reg_out: [B, 5, H, W] – предсказания регрессии (4 координаты + objectness)
        """
        shared_feat = self.shared_conv(x)
        # Применяем механизм внимания
        attn_feat = self.attention(shared_feat)
        
        # Получаем предсказания по ветвям
        cls_out = self.cls_branch(attn_feat)
        reg_out = self.reg_branch(attn_feat)
        
        return cls_out, reg_out

# Пример использования:
if __name__ == "__main__":
    # Пусть у нас есть feature map с 512 каналами и размером 20x20
    dummy_input = torch.randn(1, 512, 20, 20)
    # Допустим, у нас 80 классов (например, как в COCO)
    head = DetectionHead(in_channels=512, num_classes=80, conv_channels=256)
    cls_logits, reg_preds = head(dummy_input)
    print("Classification output shape:", cls_logits.shape)  # Ожидаем [1, 80, 20, 20]
    print("Regression output shape:", reg_preds.shape)       # Ожидаем [1, 5, 20, 20]
