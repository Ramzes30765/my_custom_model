import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List

from utils.postprocess import decode_predictions, postprocess_predictions, visualize_boxes

class MySOTAModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        head: nn.Module,
        prediction_head: nn.Module,
        topk: int = 100,
    ):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.pred_head = prediction_head
        self.topk = topk

    def forward(self, x: torch.Tensor, return_preds: bool = False, image_size: Optional[Tuple[int, int]] = None):
        """
        x: input image tensor [B, 3, H, W]
        return_preds: whether to return decoded predictions
        image_size: original image size (H, W) — required if return_preds is True
        """

        feats = self.backbone(x)              # List of feature maps
        neck_feats = self.neck(feats)         # BiFPN output [P3–P7]
        cls_feats, reg_feats = self.head(neck_feats)
        cls_outs, size_outs, offset_outs, center_outs = self.pred_head(cls_feats, reg_feats)
        
        # для отладки
        # print("mean(cls_outs[0])  :", torch.mean(cls_outs[0]))
        # print("mean(size_outs[0]) :", torch.mean(size_outs[0]))
        # print("mean(offset_outs[0]):", torch.mean(offset_outs[0]))
        # print("mean(center_outs[0]):", torch.mean(center_outs[0]))

        if not return_preds:
            return {
                "cls": cls_outs,
                "size": size_outs,
                "offset": offset_outs,
                "center": center_outs,
                "features": neck_feats
            }

        assert image_size is not None, "image_size must be provided if return_preds=True"

        raw_results = decode_predictions(cls_outs, size_outs, offset_outs, center_outs, topk=self.topk)
        processed = postprocess_predictions(raw_results, neck_feats, image_size)

        return processed  # List of (boxes, scores, classes) per image

    def predict(self, x: torch.Tensor, image_size: Tuple[int, int]):
        """
        Inference wrapper: returns decoded predictions.
        """
        return self.forward(x, return_preds=True, image_size=image_size)

    def visualize(self, image: torch.Tensor, preds, class_names=None, score_thresh=0.3):
        """
        image: Tensor [3, H, W] or numpy array
        preds: output from predict(...) → (boxes, scores, classes)
        """
        boxes, scores, classes = preds
        return visualize_boxes(image, boxes, scores, classes, class_names=class_names, score_thresh=score_thresh)