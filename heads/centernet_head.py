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

    def forward(self, feats):  # list of feature maps
        cls_out, size_out, offset_out = [], [], []
        for f in feats:
            cls_out.append(self.cls_head(f))
            size_out.append(self.size_head(f))
            offset_out.append(self.offset_head(f))
        return cls_out, size_out, offset_out