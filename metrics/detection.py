import numpy as np

def box_iou(box, boxes):
    """
    Вычисляет IoU между одним боксом и набором боксов.
    Аргументы:
      box: массив (4,) с координатами [x1, y1, x2, y2]
      boxes: массив (N, 4) с координатами боксов
    Возвращает:
      iou: массив (N,) с IoU между box и каждым боксом из boxes
    """
    # Координаты пересечения
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0, x2 - x1)
    inter_h = np.maximum(0, y2 - y1)
    inter_area = inter_w * inter_h

    # Площади боксов
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    union_area = box_area + boxes_area - inter_area + 1e-6
    iou = inter_area / union_area
    return iou

def average_precision_single(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    if pred_boxes.shape[0] == 0:
        return 0.0

    # Сортируем предсказания по оценкам от высокого к низкому.
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]

    num_preds = pred_boxes.shape[0]
    num_gts = gt_boxes.shape[0]
    gt_matched = np.zeros(num_gts, dtype=bool)

    tp = np.zeros(num_preds)  # true positives
    fp = np.zeros(num_preds)  # false positives

    for i, pred_box in enumerate(pred_boxes):
        if num_gts == 0:
            fp[i] = 1
            continue

        # Для текущего предсказания вычисляем IoU с каждым gt-боксом
        ious = box_iou(pred_box, gt_boxes)
        max_iou = np.max(ious) if ious.size > 0 else 0.0
        max_index = np.argmax(ious) if ious.size > 0 else -1

        # Если максимальное IoU больше порога и данный gt еще не использован,
        # то считаем это TP, иначе FP.
        if max_iou >= iou_threshold and not gt_matched[max_index]:
            tp[i] = 1
            gt_matched[max_index] = True
        else:
            fp[i] = 1

    # Кумулятивная сумма TP и FP используется для построения кривой precision-recall.
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    # Рассчитываем значения precision и recall на каждой точке.
    recall = cum_tp / (num_gts + 1e-6)
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    # Добавляем начальные и конечные точки к кривой (например, recall от 0 до 1).
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mprec = np.concatenate(([0.0], precision, [0.0]))

    # Выравниваем (сделаем монотонно невозрастающей) кривую precision:
    for i in range(len(mprec) - 2, -1, -1):
        mprec[i] = np.maximum(mprec[i], mprec[i + 1])

    # Находим точки, где значение recall изменяется, и интегрируем кривую.
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mprec[indices + 1])
    return ap

def compute_average_precision(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5, num_classes=None):
    if num_classes is None:
        num_classes = int(max(np.max(pred_labels), np.max(gt_labels))) + 1

    ap_per_class = []
    for c in range(num_classes):
        mask_pred = pred_labels == c
        mask_gt = gt_labels == c
        
        pred_boxes_c = pred_boxes[mask_pred]
        pred_scores_c = pred_scores[mask_pred]
        gt_boxes_c = gt_boxes[mask_gt]

        ap = average_precision_single(pred_boxes_c, pred_scores_c, gt_boxes_c, iou_threshold)
        ap_per_class.append(ap)
    mAP = np.mean(ap_per_class)
    return mAP, ap_per_class
