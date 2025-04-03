from torch import nn


class CenterHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )
        self.size_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 2, 1)
        )
        self.offset_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 2, 1)
        )
        self.center_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, 1, 1)
        )

    def forward(self, cls_feats, reg_feats):
        cls_out = [self.cls_head(f) for f in cls_feats]
        size_out = [self.size_head(f) for f in reg_feats]
        offset_out = [self.offset_head(f) for f in reg_feats]
        center_out = [self.center_head(f) for f in reg_feats]
        return cls_out, size_out, offset_out, center_out
