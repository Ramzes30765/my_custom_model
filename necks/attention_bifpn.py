import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from modules.attention_modules import CBAM, AttentionAugmentedConv, SKConv


class LateralModule(nn.Module):
    """
    Модуль для адаптации входных признаков с помощью 1x1 свёрток.
    """
    def __init__(self, in_channels_list, out_channels, use_cbam=False, ratio=16):
        super().__init__()
        self.use_cbam = use_cbam
        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels_list
        ])
        if self.use_cbam:
            self.cbam_modules = nn.ModuleList([CBAM(out_channels, ratio) for _ in in_channels_list])
    
    def forward(self, features):
        # features — список входных карт признаков
        outs = []
        for i, x in enumerate(features):
            x = self.convs[i](x)
            if self.use_cbam:
                x = self.cbam_modules[i](x)
            outs.append(x)
        return outs


class FusionModule(nn.Module):
    """
    Модуль объединения (fusion) признаков в BiFPN.
    Реализует топ-даун и bottom-up пути с возможностью включения:
      - CBAM в топ-даун пути (use_td_cbam)
      - SKConv и AttentionAugmentedConv в bottom-up пути (use_skconv и use_aac)
    
    Принимает список признаков [p2, p3, p4, p5] и возвращает обработанный список.
    """
    def __init__(
        self,
        out_channels,
        use_td_cbam=False,
        use_skconv=False,
        use_aac=False,
        # cbam params
        ratio=16,
        # skconv params
        M=2, r=16,
        # aac params
        dqk=32,
        dv=4,
        Nh=4,
        
        ):
        super().__init__()
        self.out_channels = out_channels
        self.use_td_cbam = use_td_cbam
        self.ratio = ratio
        self.use_skconv = use_skconv
        self.M = M
        self.r = r
        self.use_aac = use_aac
        self.dqk = dqk
        self.dv = dv
        self.Nh = Nh
        
        # Для топ-даун слияния (p4, p3, p2): три обучаемых набора весов
        self.td_weights = nn.ParameterList([nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(3)])
        if self.use_td_cbam:
            self.td_cbam_modules = nn.ModuleList([CBAM(out_channels, self.ratio) for _ in range(3)])
        
        # Для bottom-up слияния (p3, p4, p5): три обучаемых набора весов
        self.bu_weights = nn.ParameterList([nn.Parameter(torch.ones(2, dtype=torch.float32)) for _ in range(3)])
        
        # Для p2 bottom-up – стандартная свёртка 3x3
        self.conv_p2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Для bottom-up fusion на уровнях p3, p4, p5 можно применить SKConv и/или AAC.
        if self.use_skconv:
            self.skconv_modules = nn.ModuleList([
                SKConv(out_channels, M=self.M, r=self.r) for _ in range(3)
            ])
        if self.use_aac:
            self.aac_modules = nn.ModuleList([
                AttentionAugmentedConv(out_channels, out_channels, kernel_size=3, dqk=self.dqk, dv=self.dv, Nh=self.Nh) for _ in range(3)
            ])
    
    def _weighted_sum(self, x, y, weights):
        # Вычисляем softmax по весам и возвращаем их взвешенную сумму
        w = torch.softmax(weights, dim=0)
        return w[0] * x + w[1] * y
    
    def forward(self, features):
        """
        Args:
        features: список из 4-х признаков [p2, p3, p4, p5]
        """
        # --- Топ-даун путь ---
        # p5: оставляем без изменений
        td_p5 = features[3]

        # p4: объединяем исходный p4 и upsampled p5
        up_p5 = F.interpolate(td_p5, size=features[2].shape[2:], mode='nearest')
        td_p4 = self._weighted_sum(features[2], up_p5, self.td_weights[0])
        if self.use_td_cbam:
            td_p4 = self.td_cbam_modules[0](td_p4)

        # p3: объединяем исходный p3 и upsampled p4
        up_td_p4 = F.interpolate(td_p4, size=features[1].shape[2:], mode='nearest')
        td_p3 = self._weighted_sum(features[1], up_td_p4, self.td_weights[1])
        if self.use_td_cbam:
            td_p3 = self.td_cbam_modules[1](td_p3)

        # p2: объединяем исходный p2 и upsampled p3
        up_td_p3 = F.interpolate(td_p3, size=features[0].shape[2:], mode='nearest')
        td_p2 = self._weighted_sum(features[0], up_td_p3, self.td_weights[2])
        if self.use_td_cbam:
            td_p2 = self.td_cbam_modules[2](td_p2)

        # Сохраняем результаты топ-даун пути
        td_features = [td_p2, td_p3, td_p4, td_p5]

        # --- Bottom-up путь ---
        # p2: обрабатываем стандартной свёрткой 3x3
        bu_p2 = self.conv_p2(td_p2)

        # p3: объединяем td_p3 и upsampled bu_p2
        up_bu_p2 = F.interpolate(bu_p2, size=td_p3.shape[2:], mode='nearest')
        bu_p3 = self._weighted_sum(td_p3, up_bu_p2, self.bu_weights[2])
        if self.use_skconv:
            bu_p3 = self.skconv_modules[0](bu_p3)
        if self.use_aac:
            bu_p3 = self.aac_modules[0](bu_p3)

        # p4: объединяем td_p4 и upsampled bu_p3
        up_bu_p3 = F.interpolate(bu_p3, size=td_p4.shape[2:], mode='nearest')
        bu_p4 = self._weighted_sum(td_p4, up_bu_p3, self.bu_weights[1])
        if self.use_skconv:
            bu_p4 = self.skconv_modules[1](bu_p4)
        if self.use_aac:
            bu_p4 = self.aac_modules[1](bu_p4)

        # p5: объединяем td_p5 и upsampled bu_p4
        up_bu_p4 = F.interpolate(bu_p4, size=td_p5.shape[2:], mode='nearest')
        bu_p5 = self._weighted_sum(td_p5, up_bu_p4, self.bu_weights[0])
        if self.use_skconv:
            bu_p5 = self.skconv_modules[2](bu_p5)
        if self.use_aac:
            bu_p5 = self.aac_modules[2](bu_p5)

        # Собираем итоговые признаки bottom-up
        bu_features = [bu_p2, bu_p3, bu_p4, bu_p5]
        return bu_features


class BiFPN(nn.Module):
    """
    Основной модуль BiFPN, который объединяет backbone, lateral- и fusion модули.
    Позволяет задавать количество слоёв BiFPN и включать/выключать attention-модули.
    
    Параметры:
        ....
      - use_lateral_cbam: применять CBAM после lateral свёрток
      - use_td_cbam: применять CBAM в топ-даун пути fusion
      - use_skconv: применять SKConv в bottom-up пути fusion
      - use_aac: применять AttentionAugmentedConv в bottom-up пути fusion
    """
    def __init__(self, in_channels_list, out_channels=256, num_layers=1,
                 use_lateral_cbam=False, use_td_cbam=False, use_skconv=False, use_aac=False):
        super().__init__()
        self.lateral_module = LateralModule(in_channels_list, out_channels, use_cbam=use_lateral_cbam)
        self.fusion_module = FusionModule(out_channels,
                                          use_td_cbam=use_td_cbam,
                                          use_skconv=use_skconv,
                                          use_aac=use_aac
                                          )
        self.num_layers = num_layers
    
    def forward(self, features):
        lateral_feats = self.lateral_module(features[:4])

        for _ in range(self.num_layers):
            lateral_feats = self.fusion_module(lateral_feats)
            
        # Генерация P6 и P7 из P5
        p5 = lateral_feats[-1]                            # [B, C, H/32, W/32]
        p6 = F.max_pool2d(p5, kernel_size=2, stride=2)  # → [B, C, H/64, W/64]
        p7 = F.max_pool2d(p6, kernel_size=2, stride=2)  # → [B, C, H/128, W/128]

        # Теперь список: [P2, P3, P4, P5, P6, P7]
        full_feats = lateral_feats + [p6, p7]
        return tuple(full_feats[1:])