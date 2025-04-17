import torch

def compute_det_metrics(preds, targets, metric_obj):
    """
    Обновляет метрику по предсказаниям и таргетам.
    Предсказания и таргеты должны быть в одной системе координат (обычно original_size).

    preds: Tuple[boxes, scores, labels] — результат model.predict()
    targets: Dict, должен содержать "original_targets" с ключами "boxes" и "labels"
    metric_obj: экземпляр torchmetrics.Metric (например, DetectionMAP)
    """
    boxes, scores, labels = preds

    metric_obj.update(
        preds=[{
            "boxes": boxes.detach().cpu(),
            "scores": scores.detach().cpu(),
            "labels": labels.detach().cpu().int(),
        }],
        targets=[{
            "boxes": targets["original_targets"]["boxes"].detach().cpu(),
            "labels": targets["original_targets"]["labels"].detach().cpu().int(),
        }]
    )