import torch
from torch import nn
import torch.nn.functional as F
import timm

# Предполагается, что эти модули уже реализованы:
from modules.attention_modules import CBAM, AAC_CBAM_Sequential, AAC_CBAM_Parallel

def get_norm_layer(num_features, norm_type, num_groups):
    if norm_type == "batch":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "layer":
        # Используем GroupNorm с 1 группой как аналог LayerNorm для conv-тензоров
        return nn.GroupNorm(1, num_features)
    elif norm_type == "group":
        return nn.GroupNorm(num_groups, num_features)
    else:
        return nn.Identity()

class BiFPN(nn.Module):
    def __init__(
        self,
        backbone_name='resnet50',
        out_channels=256,
        num_layers=1,
        attn_block='none',  # Возможные значения: "none", "cbam", "aac_cbam_sequential", "aac_cbam_parallel"
        use_attention_lateral=False,  # применять внимание после латеральных свёрток
        use_attention_fusion=False,     # применять внимание после объединяющих свёрток
        d_qk=64, # размерность веторов query и key
        dv=8, # размерность ветора value
        Nh=8, # число голов для FlexAttention
        norm_type="batch",    # тип нормализации: "batch", "layer", "group", "none"
        dropout_rate=0.0,     # dropout, например 0.1 или 0.2
        num_groups=32         # число групп для групповой нормализации, если используется "group"
    ):
        super().__init__()
        self.attn_block = attn_block
        self.use_attention_lateral = use_attention_lateral
        self.use_attention_fusion = use_attention_fusion
        
        # Инициализация бэкбона через timm
        self.backbone = timm.create_model(
            backbone_name,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            pretrained=True
        )
        self.in_channels_list = self.backbone.feature_info.channels()
        
        # Латеральные свёртки
        self.lateral_p2 = nn.Conv2d(self.in_channels_list[0], out_channels, 1)
        self.lateral_p3 = nn.Conv2d(self.in_channels_list[1], out_channels, 1)
        self.lateral_p4 = nn.Conv2d(self.in_channels_list[2], out_channels, 1)
        self.lateral_p5 = nn.Conv2d(self.in_channels_list[3], out_channels, 1)
        
        # Нормализация и dropout после латеральных свёрток
        self.norm_lateral_p2 = get_norm_layer(out_channels, norm_type, num_groups)
        self.norm_lateral_p3 = get_norm_layer(out_channels, norm_type, num_groups)
        self.norm_lateral_p4 = get_norm_layer(out_channels, norm_type, num_groups)
        self.norm_lateral_p5 = get_norm_layer(out_channels, norm_type, num_groups)
        
        self.dropout_lateral_p2 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout_lateral_p3 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout_lateral_p4 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout_lateral_p5 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Применяем блок внимания после латеральных свёрток (если включено)
        if self.use_attention_lateral:
            if self.attn_block == 'cbam':
                self.attn_lateral_p2 = CBAM(out_channels)
                self.attn_lateral_p3 = CBAM(out_channels)
                self.attn_lateral_p4 = CBAM(out_channels)
                self.attn_lateral_p5 = CBAM(out_channels)
            elif self.attn_block == 'aac_cbam_sequential':
                self.attn_lateral_p2 = AAC_CBAM_Sequential(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_lateral_p3 = AAC_CBAM_Sequential(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_lateral_p4 = AAC_CBAM_Sequential(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_lateral_p5 = AAC_CBAM_Sequential(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
            elif self.attn_block == 'aac_cbam_parallel':
                self.attn_lateral_p2 = AAC_CBAM_Parallel(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_lateral_p3 = AAC_CBAM_Parallel(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_lateral_p4 = AAC_CBAM_Parallel(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_lateral_p5 = AAC_CBAM_Parallel(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
            else:
                self.attn_lateral_p2 = self.attn_lateral_p3 = self.attn_lateral_p4 = self.attn_lateral_p5 = None
        else:
            self.attn_lateral_p2 = self.attn_lateral_p3 = self.attn_lateral_p4 = self.attn_lateral_p5 = None
        
        # BiFPN параметры
        self.num_layers = num_layers
        self.weights_top_down = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(3)
        ])
        self.weights_bottom_up = nn.ParameterList([
            nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(3)
        ])
        
        # Свёртки после объединения (fusion)
        self.conv_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_p5 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Нормализация и dropout после fusion свёрток
        self.norm_fusion_p2 = get_norm_layer(out_channels, norm_type, num_groups)
        self.norm_fusion_p3 = get_norm_layer(out_channels, norm_type, num_groups)
        self.norm_fusion_p4 = get_norm_layer(out_channels, norm_type, num_groups)
        self.norm_fusion_p5 = get_norm_layer(out_channels, norm_type, num_groups)
        
        self.dropout_fusion_p2 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout_fusion_p3 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout_fusion_p4 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.dropout_fusion_p5 = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Применяем блок внимания после fusion свёрток (если включено)
        if self.use_attention_fusion:
            if self.attn_block == 'cbam':
                self.attn_fusion_p2 = CBAM(out_channels)
                self.attn_fusion_p3 = CBAM(out_channels)
                self.attn_fusion_p4 = CBAM(out_channels)
                self.attn_fusion_p5 = CBAM(out_channels)
            elif self.attn_block == 'aac_cbam_sequential':
                self.attn_fusion_p2 = AAC_CBAM_Sequential(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_fusion_p3 = AAC_CBAM_Sequential(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_fusion_p4 = AAC_CBAM_Sequential(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_fusion_p5 = AAC_CBAM_Sequential(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
            elif self.attn_block == 'aac_cbam_parallel':
                self.attn_fusion_p2 = AAC_CBAM_Parallel(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_fusion_p3 = AAC_CBAM_Parallel(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_fusion_p4 = AAC_CBAM_Parallel(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
                self.attn_fusion_p5 = AAC_CBAM_Parallel(out_channels, out_channels, kernel_size=3, dk=64, dv=8, Nh=8, use_flash=True)
            else:
                self.attn_fusion_p2 = self.attn_fusion_p3 = self.attn_fusion_p4 = self.attn_fusion_p5 = None
        else:
            self.attn_fusion_p2 = self.attn_fusion_p3 = self.attn_fusion_p4 = self.attn_fusion_p5 = None

    def forward(self, x):
        # Извлекаем признаки из бэкбона
        features = self.backbone(x)
        
        # Латеральные признаки с нормализацией и dropout
        p2 = self.dropout_lateral_p2(self.norm_lateral_p2(self.lateral_p2(features[0])))
        p3 = self.dropout_lateral_p3(self.norm_lateral_p3(self.lateral_p3(features[1])))
        p4 = self.dropout_lateral_p4(self.norm_lateral_p4(self.lateral_p4(features[2])))
        p5 = self.dropout_lateral_p5(self.norm_lateral_p5(self.lateral_p5(features[3])))
        
        # Применяем блок внимания после латеральных свёрток (если включено)
        if self.use_attention_lateral:
            if self.attn_lateral_p2 is not None:
                p2 = self.attn_lateral_p2(p2)
            if self.attn_lateral_p3 is not None:
                p3 = self.attn_lateral_p3(p3)
            if self.attn_lateral_p4 is not None:
                p4 = self.attn_lateral_p4(p4)
            if self.attn_lateral_p5 is not None:
                p5 = self.attn_lateral_p5(p5)
        
        # BiFPN обработка
        for _ in range(self.num_layers):
            # Top-down pathway
            p5_td = p5
            p4_td = self._weighted_sum(
                p4,
                F.interpolate(p5_td, size=p4.shape[2:], mode='nearest'),
                self.weights_top_down[0]
            )
            p3_td = self._weighted_sum(
                p3,
                F.interpolate(p4_td, size=p3.shape[2:], mode='nearest'),
                self.weights_top_down[1]
            )
            p2_td = self._weighted_sum(
                p2,
                F.interpolate(p3_td, size=p2.shape[2:], mode='nearest'),
                self.weights_top_down[2]
            )
            
            # Bottom-up pathway с fusion свёртками, нормализацией и dropout
            p2_out = self.dropout_fusion_p2(self.norm_fusion_p2(self.conv_p2(p2_td)))
            p3_out = self.dropout_fusion_p3(self.norm_fusion_p3(
                self.conv_p3(self._weighted_sum(
                    p3_td,
                    F.interpolate(p2_out, size=p3_td.shape[2:], mode='nearest'),
                    self.weights_bottom_up[2]
                ))
            ))
            p4_out = self.dropout_fusion_p4(self.norm_fusion_p4(
                self.conv_p4(self._weighted_sum(
                    p4_td,
                    F.interpolate(p3_out, size=p4_td.shape[2:], mode='nearest'),
                    self.weights_bottom_up[1]
                ))
            ))
            p5_out = self.dropout_fusion_p5(self.norm_fusion_p5(
                self.conv_p5(self._weighted_sum(
                    p5_td,
                    F.interpolate(p4_out, size=p5_td.shape[2:], mode='nearest'),
                    self.weights_bottom_up[0]
                ))
            ))
            
            # Применяем блок внимания после fusion свёрток (если включено)
            if self.use_attention_fusion:
                if self.attn_fusion_p2 is not None:
                    p2_out = self.attn_fusion_p2(p2_out)
                if self.attn_fusion_p3 is not None:
                    p3_out = self.attn_fusion_p3(p3_out)
                if self.attn_fusion_p4 is not None:
                    p4_out = self.attn_fusion_p4(p4_out)
                if self.attn_fusion_p5 is not None:
                    p5_out = self.attn_fusion_p5(p5_out)
            
            p2, p3, p4, p5 = p2_out, p3_out, p4_out, p5_out
        
        return p2_out, p3_out, p4_out, p5_out

    def _weighted_sum(self, x, y, weights):
        w = torch.softmax(weights, dim=0)
        return w[0] * x + w[1] * y
