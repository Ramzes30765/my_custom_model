from torch import nn

from neck.fpn import FPN


class PANet(nn.Module):
    def __init__(self, backbone_name='resnet50', out_channels=256):
        super().__init__()
        # Инициализация FPN
        self.fpn = FPN(backbone_name, out_channels)
        
        # Дополнительные свёртки для Bottom-Up Pathway
        self.down_conv_p2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.down_conv_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.down_conv_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        # Получаем признаки из FPN
        p2, p3, p4, p5 = self.fpn(x)
        
        # Bottom-Up Pathway (передача признаков вниз)
        n2 = p2
        n3 = p3 + self.down_conv_p2(n2)
        n4 = p4 + self.down_conv_p3(n3)
        n5 = p5 + self.down_conv_p4(n4)
        
        return n2, n3, n4, n5