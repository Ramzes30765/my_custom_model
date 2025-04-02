import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention


class CBAM(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        # Канальный модуль
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels//ratio, 1),
            nn.ReLU(),
            nn.Conv2d(channels//ratio, channels, 1),
            nn.Sigmoid()
        )
        # Пространственный модуль
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Канальный аттеншн
        channel_mask = self.channel_att(x)
        x = x * channel_mask
        # Пространственный аттеншн
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_mask = self.spatial_att(spatial_input)
        x = x * spatial_mask
        
        return x


class AttentionAugmentedConv(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size=3, 
        dqk=32,    # общая размерность для Q и K (должна быть кратна числу голов)
        dv=4,     # размерность на одну голову для V
        Nh=4,     # число голов
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        
        self.fc_q = nn.Linear(in_channels, dqk)
        self.fc_k = nn.Linear(in_channels, dqk)
        # Проекция для V теперь выдаёт вектор размерности (Nh * dv)
        self.fc_v = nn.Linear(in_channels, Nh * dv)
        # Выходной слой принимает объединённый вектор размерности (Nh * dv)
        self.out_proj = nn.Linear(Nh * dv, in_channels)
        self.Nh = Nh

    def forward(self, x):
        
        batch, C, H, W = x.shape
        L = H * W
        
        # Преобразуем вход из (B, C, H, W) в (B, L, C)
        x_flat = x.view(batch, C, L).permute(0, 2, 1)
        
        # Вычисляем Q, K, V
        q = self.fc_q(x_flat)  # (B, L, dk)
        k = self.fc_k(x_flat)  # (B, L, dk)
        v = self.fc_v(x_flat)  # (B, L, Nh*dv)
        
        # Разбиваем Q и K на Nh голов
        dqk_per_head = q.shape[-1] // self.Nh
        q = q.view(batch, L, self.Nh, dqk_per_head).permute(0, 2, 1, 3)  # (B, Nh, L, dk_per_head)
        k = k.view(batch, L, self.Nh, dqk_per_head).permute(0, 2, 1, 3)  # (B, Nh, L, dk_per_head)
        
        # Разбиваем V на Nh голов; каждая голова имеет размерность dv
        v = v.view(batch, L, self.Nh, -1).permute(0, 2, 1, 3)  # (B, Nh, L, dv)
        
        # Вычисляем scaled dot-product attention через flex_attention
        scale = q.size(-1) ** -0.5
        attn_output = flex_attention(q, k, v, scale=scale)
        # attn_output имеет форму (B, Nh, L, dv)
        
        # Объединяем головы: (B, Nh, L, dv) -> (B, L, Nh*dv)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch, L, self.Nh * v.size(-1))
        
        # Применяем выходной проектор и возвращаем форму к изображению: (B, L, in_channels) -> (B, in_channels, H, W)
        out = self.out_proj(attn_output).view(batch, H, W, C).permute(0, 3, 1, 2)
        
        # Резидуальное соединение и свёрточная обработка
        return self.conv(out + x)


class SKConv(nn.Module):
    def __init__(self, in_channels, M=2, r=16):
        super().__init__()
        self.M = M
        self.out_channels = in_channels
        self.convs = nn.ModuleList()
        for i in range(M):
            kernel_size = 3 + 2 * i
            padding = 1 + i
            self.convs.append(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding))
        # Снижаем размерность: (batch, C) -> (batch, C//r)
        self.fc = nn.Linear(in_channels, in_channels // r)
        # Для каждой ветви отдельный линейный слой, возвращающий вес для каждого канала
        self.fcs = nn.ModuleList()
        for i in range(M):
            self.fcs.append(nn.Linear(in_channels // r, in_channels))
        # Будем применять softmax по оси ветвей (M)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        batch, C, H, W = x.size()
        # Применяем M свёрток
        feats = [conv(x) for conv in self.convs]  # каждый имеет форму (batch, C, H, W)
        feats = torch.stack(feats, dim=1)         # (batch, M, C, H, W)
        
        # Фьюжн признаков: суммируем по ветвям и применяем глобальное среднее по пространству
        U = feats.sum(dim=1)          # (batch, C, H, W)
        U = U.mean(dim=[2, 3])        # (batch, C)
        z = self.fc(U)                # (batch, C//r)
        
        # Вычисляем веса для каждой ветви
        # Здесь формируем список тензоров размера (batch, 1, C)
        weights = [fc(z).unsqueeze(1) for fc in self.fcs]
        weights = torch.cat(weights, dim=1)   # (batch, M, C)
        weights = self.softmax(weights)       # нормализуем по ветвям (M)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (batch, M, C, 1, 1)
        
        # Взвешиваем признаки по ветвям и суммируем
        V = (feats * weights).sum(dim=1)   # (batch, C, H, W)
        return V