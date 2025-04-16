import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DetectionMetricsWrapper:
    def __init__(self, iou_type="bbox", compute_per_class=True):
        """
        Обёртка над torchmetrics.detection.MeanAveragePrecision
        Поддерживает:
            - COCO-style и VOC-style (iou_type="bbox")
            - Пер-классные метрики
        """
        self.metric = MeanAveragePrecision(iou_type=iou_type, class_metrics=compute_per_class)

    def update(self, preds, targets):
        """
        Аргументы:
            preds: List[Dict] — [{"boxes": [N, 4], "scores": [N], "labels": [N]}]
            targets: List[Dict] — [{"boxes": [M, 4], "labels": [M]}]
        """
        self.metric.update(preds, targets)

    def compute(self):
        """
        Возвращает:
            dict с метриками: {'map', 'map_50', 'map_75', 'map_per_class', 'mar', ...}
        """
        return self.metric.compute()

    def reset(self):
        self.metric.reset()