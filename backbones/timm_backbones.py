import timm
import torch.nn as nn

class BackboneWrapper(nn.Module):
    def __init__(self, name='resnet50', pretrained=True, out_indices=(-4, -3, -2, -1), checkpoint_path=None):
        super().__init__()
        self.model = timm.create_model(
            name,
            features_only=True,
            out_indices=out_indices,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path
        )

    def forward(self, x):
        return self.model(x)
