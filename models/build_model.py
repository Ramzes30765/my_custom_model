from backbones.timm_backbones import BackboneWrapper
from necks.attention_bifpn import BiFPN
from heads.dynamic_head import DynamicHead
from heads.prediction_head import CenterHead
from sota_model import MySOTAModel


def build_model(num_classes: int) -> MySOTAModel:
    # Backbone: ResNet50
    backbone = BackboneWrapper(name="resnet50", pretrained=True)

    # Neck: BiFPN
    neck = BiFPN(
        backbone_name="resnet50",
        out_channels=256,
        num_layers=2,
        attn_block="cbam",
        use_attention_lateral=True,
        use_attention_fusion=True
    )

    # Dynamic Head
    head = DynamicHead(
        in_channels=256,
        num_levels=4,
        num_blocks=6,
        num_tasks=2
    )

    # Prediction Head: Center-based
    pred_head = CenterHead(in_channels=256, num_classes=num_classes)

    return MySOTAModel(
        backbone=backbone,
        neck=neck,
        head=head,
        prediction_head=pred_head,
        topk=100
    )
