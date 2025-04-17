import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List

from utils.postprocess import decode_predictions, postprocess_predictions
from utils.vizualization import visualize_boxes

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

    def forward(self, x: torch.Tensor, return_preds: bool = False,
                image_size: Optional[Tuple[int, int]] = None,
                topk: int = 100, score_thresh: float = 0.3):
        """
        x: input image tensor [B, 3, H, W]
        return_preds: whether to return decoded predictions
        image_size: original image size (H, W) — required if return_preds is True
        """

        feats = self.backbone(x)              # List of feature maps
        neck_feats = self.neck(feats)         # BiFPN output [P3–P7]
        cls_feats, reg_feats = self.head(neck_feats)
        cls_outs, size_outs, offset_outs, center_outs = self.pred_head(cls_feats, reg_feats)

        if not return_preds:
            return {
                "cls": cls_outs,
                "size": size_outs,
                "offset": offset_outs,
                "center": center_outs,
                "features": neck_feats
            }

        assert image_size is not None, "image_size must be provided if return_preds=True"

        raw_results = decode_predictions(cls_outs, size_outs, offset_outs, center_outs, topk=topk)
        processed = postprocess_predictions(raw_results, neck_feats, image_size, score_thresh=score_thresh)

        return processed  # List of (boxes, scores, classes) per image

    def predict(self, x, image_size=None, original_size=None, topk=100, score_thresh=0.3):
        """
        Универсальный предикт: работает как с батчем, так и с одной картинкой.
        """
        is_single_image = False
        if x.ndim == 3:
            x = x.unsqueeze(0)  # [3, H, W] → [1, 3, H, W]
            is_single_image = True

        preds = self.forward(x, return_preds=True, image_size=image_size, topk=topk, score_thresh=score_thresh)

        # Масштабируем, если нужно
        if original_size is not None:
            orig_h, orig_w = original_size
            input_h, input_w = image_size
            scale_x = orig_w / input_w
            scale_y = orig_h / input_h

            scaled_preds = []
            for boxes, scores, classes in preds:
                boxes = boxes.clone()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                scaled_preds.append((boxes, scores, classes))
            preds = scaled_preds

        if is_single_image:
            return preds[0]  # вернём один результат
        return preds  # список из B элементов

    def visualize(self, image: torch.Tensor, preds, class_names=None):
        """
        image: Tensor [3, H, W] or numpy array
        preds: output from predict(...) → (boxes, scores, classes)
        """
        boxes, scores, classes = preds
        return visualize_boxes(image, boxes, scores, classes, class_names=class_names)