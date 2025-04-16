from backbones.timm_backbones import BackboneWrapper
from necks.attention_bifpn import BiFPN
from heads.dynamic_head import DynamicHead
from heads.prediction_head import CenterHead
from .sota_model import MySOTAModel


def build_model(num_classes: int, backbone_name: str = 'resnet18', pretrained:bool = True) -> MySOTAModel:

    backbone = BackboneWrapper(
        name=backbone_name,
        pretrained=pretrained
    )
    in_channels_list = backbone.model.feature_info.channels()
    # Neck: BiFPN
    neck = BiFPN(
        in_channels_list=in_channels_list,
        out_channels=256,
        num_layers=2,
        use_lateral_cbam=False,
        use_td_cbam=False,
        use_skconv=False,
        use_aac=False
    )

    # Dynamic Head
    head = DynamicHead(
        in_channels=256,
        num_levels=5,
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
