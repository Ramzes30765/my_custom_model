import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List

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

    def forward(self, x: torch.Tensor, image_size: Optional[Tuple[int, int]] = None):
        """
        x: input image tensor [B, 3, H, W]
        image_size: original image size (H, W) — required if return_preds is True
        return_preds: whether to return decoded predictions
        """

        feats = self.backbone(x)              # List of feature maps
        neck_feats = self.neck(feats)         # BiFPN output [P3–P7]
        cls_feats, reg_feats = self.head(neck_feats)
        cls_outs, size_outs, offset_outs, center_outs = self.pred_head(cls_feats, reg_feats)
        
        return {
            "cls": cls_outs,
            "size": size_outs,
            "offset": offset_outs,
            "center": center_outs,
            "features": neck_feats
        }

    @torch.no_grad()
    def predict(self, x, image_size=None, original_size=None, topk=100, score_thresh=0.3, nms_iou=0.5):
        """
        Универсальный предикт: работает как с батчем, так и с одной картинкой.
        """
        is_single_image = False
        if x.ndim == 3:
            x = x.unsqueeze(0)   # [3,H,W] -> [1,3,H,W]
            is_single_image = True
        elif x.ndim == 4 and x.size(0) == 1:
            is_single_image = True

        assert image_size is not None, "image_size must be provided if return_preds=True"
        
        preds = self.forward(x, image_size=image_size)
        decoded_preds = self.pred_head.decode_preds(
            preds['cls'],
            preds['size'],
            preds['offset'],
            preds['center'],
            preds['features'],
            image_size,
            topk,
            score_thresh,
            nms_iou
        )
        # Масштабируем, если нужно
        if original_size is not None:
            if isinstance(original_size, tuple):
                # один размер на весь батч
                Hori, Wori = original_size
                Hin, Win = image_size
                sx, sy = Wori / Win, Hori / Hin
                scaled = []
                for boxes, scores, labels in decoded_preds:
                    if boxes.numel() > 0:
                        boxes = boxes.clone()
                        boxes[:, [0, 2]] *= sx
                        boxes[:, [1, 3]] *= sy
                    scaled.append((boxes, scores, labels))
                out = scaled
            else:
                # список размеров по каждому элементу батча
                Hin, Win = image_size
                scaled = []
                for (boxes, scores, labels), (Hori, Wori) in zip(out, original_size):
                    sx, sy = Wori / Win, Hori / Hin
                    if boxes.numel() > 0:
                        boxes = boxes.clone()
                        boxes[:, [0, 2]] *= sx
                        boxes[:, [1, 3]] *= sy
                    scaled.append((boxes, scores, labels))
                out = scaled

        if is_single_image:
            return out[0]  # вернём один результат
        return out  # список из B элементов

    def visualize(self, image: torch.Tensor, preds, class_names=None):
        """
        image: Tensor [3, H, W] or numpy array
        preds: output from predict(...) → (boxes, scores, classes)
        """
        boxes, scores, classes = preds
        return visualize_boxes(image, boxes, scores, classes, class_names=class_names)