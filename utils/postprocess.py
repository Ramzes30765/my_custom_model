import torch
import cv2
import numpy as np
from torchvision.ops import batched_nms

from utils.preprocess import get_strides_from_feature_maps

def split_topk_per_level(level_shapes, topk):
    """
    level_shapes: список кортежей (C, H, W) по уровням
    Возвращает список k_per_level, сумма == topk.
    """
    import math
    sizes = [C * H * W for (C, H, W) in level_shapes]
    total = max(1, sum(sizes))
    raw = [topk * s / total for s in sizes]

    k_floor = [int(math.floor(x)) for x in raw]
    remainder = topk - sum(k_floor)

    # распределяем остаток уровням с наибольшей дробной частью
    frac = [(x - f, i) for i, (x, f) in enumerate(zip(raw, k_floor))]
    frac.sort(reverse=True)  # по убыванию дробной части
    for j in range(remainder):
        i = frac[j][1]
        k_floor[i] += 1

    # гарантируем минимум 1, если есть кандидаты на уровне
    for i, s in enumerate(sizes):
        if s > 0:
            k_floor[i] = max(1, min(k_floor[i], s))
        else:
            k_floor[i] = 0
    return k_floor

def decode_predictions(cls_outputs, size_outputs, offset_outputs, center_outputs, topk=100):
    results = []
    
    level_shapes = [m.shape[1:4] for m in cls_outputs]
    k_per_level = split_topk_per_level(level_shapes, topk)
    
    for cls_map, size_map, offset_map, center_map, k in zip(
        cls_outputs, size_outputs, offset_outputs, center_outputs, k_per_level
        ):
        B, C, H, W = cls_map.shape

        # 1) вероятности и «центровая» маска
        scores_map = torch.sigmoid(cls_map) * torch.sigmoid(center_map)  # [B,C,H,W]

        # 2) top-K на уровне карты
        flat = scores_map.view(B, -1)                                    # [B, C*H*W]
        k = min(k, flat.shape[1])
        scores, inds = torch.topk(flat, k=k, dim=1)                      # [B,k]

        # 3) восстановление класса и позиции клетки
        classes = (inds // (H * W)).to(torch.int64)                      # [B,k]
        pix = inds % (H * W)                                             # [B,k]
        xs = (pix % W).float()
        ys = (pix // W).float()

        # 4) выборка size/offset, расчёт боксов
        size_map = size_map.permute(0, 2, 3, 1).reshape(B, -1, 2)        # [B,H*W,2]
        offset_map = offset_map.permute(0, 2, 3, 1).reshape(B, -1, 2)    # [B,H*W,2]
        gather = pix.unsqueeze(-1).repeat(1, k, 2)                       # [B,k,2]
        sizes   = torch.gather(size_map,   1, gather)                    # [B,k,2]
        offsets = torch.gather(offset_map, 1, gather)                    # [B,k,2]

        # гарантируем неотрицательные размеры
        w = sizes[..., 0].relu()
        h = sizes[..., 1].relu()

        cx = xs + offsets[..., 0]
        cy = ys + offsets[..., 1]

        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)                    # [B,k,4]
        results.append((boxes, scores, classes))
    return results

def postprocess_predictions(results, features, image_size, score_thresh, nms_iou=0.5, keep_topk=None):
    
    strides = get_strides_from_feature_maps(image_size, features)
    B = results[0][0].shape[0]
    batch_results = []
    
    for b in range(B):
        per_img_boxes, per_img_scores, per_img_classes = [], [], []

        # 1) собираем ВСЕ уровни без порога
        for (boxes, scores, classes), stride in zip(results, strides):
            boxes_b   = boxes[b] * stride       # в пикселях
            scores_b  = scores[b]
            classes_b = classes[b].to(torch.int64)
            per_img_boxes.append(boxes_b)
            per_img_scores.append(scores_b)
            per_img_classes.append(classes_b)

        boxes_all   = torch.cat(per_img_boxes,   dim=0)
        scores_all  = torch.cat(per_img_scores,  dim=0)
        classes_all = torch.cat(per_img_classes, dim=0)

        # 2) единый threshold
        keep = scores_all > score_thresh
        boxes_all   = boxes_all[keep]
        scores_all  = scores_all[keep]
        classes_all = classes_all[keep]

        # 3) NMS (class-aware)
        if boxes_all.numel() > 0:
            keep_idx = batched_nms(boxes_all, scores_all, classes_all, iou_threshold=nms_iou)
            boxes_all   = boxes_all[keep_idx]
            scores_all  = scores_all[keep_idx]
            classes_all = classes_all[keep_idx]

        # 4) (опционально) глобальный top-K после NMS
        if keep_topk is not None and boxes_all.size(0) > keep_topk:
            vals, inds = torch.topk(scores_all, k=keep_topk, dim=0)
            boxes_all   = boxes_all[inds]
            scores_all  = vals
            classes_all = classes_all[inds]

        # 5) клип по границам
        H, W = image_size
        boxes_all[:, 0].clamp_(0, W); boxes_all[:, 2].clamp_(0, W)
        boxes_all[:, 1].clamp_(0, H); boxes_all[:, 3].clamp_(0, H)

        batch_results.append((boxes_all, scores_all, classes_all))

    return batch_results

def decode_predictions_global_topk(
    cls_outputs,
    size_outputs,
    offset_outputs,
    center_outputs,
    features,
    image_size,
    topk=100,
    score_thresh=0.25,
    nms_iou=None
):
    """
    Глобальный top-K по всем уровням.
    Возвращает список длины B: (boxes_xyxy[Ni,4], scores[Ni], labels[Ni]).
    boxes уже в ПИКСЕЛЯХ image_size.
    """
    strides = get_strides_from_feature_maps(image_size, features)  # уже есть в файле
    L = len(cls_outputs)
    B = cls_outputs[0].shape[0]

    batch_results = []

    for b in range(B):
        # 1) соберём плоские score-векторы по всем уровням для картинки b
        level_scores = []
        level_shapes = []   # (C,H,W) на уровень
        level_offsets = []  # префиксные суммы длины, чтобы маппить глобальный индекс -> уровень
        cum = 0

        for l in range(L):
            cls_map    = torch.sigmoid(cls_outputs[l][b])    # [C,H,W]
            center_map = torch.sigmoid(center_outputs[l][b]) # [1,H,W] или [C,H,W]
            scores_map = cls_map * center_map                # [C,H,W]
            C, H, W = scores_map.shape
            flat = scores_map.view(-1)                       # [C*H*W]
            level_scores.append(flat)
            level_shapes.append((C, H, W))
            level_offsets.append(cum)
            cum += flat.numel()

        if cum == 0:
            batch_results.append((
                torch.empty((0, 4), device=cls_outputs[0].device),
                torch.empty((0,),  device=cls_outputs[0].device),
                torch.empty((0,),  device=cls_outputs[0].device, dtype=torch.long),
            ))
            continue

        scores_all = torch.cat(level_scores, dim=0)          # [N]
        K = min(topk, scores_all.numel())

        # 2) глобальный top-K
        scores_k, inds_k = torch.topk(scores_all, k=K, dim=0)  # [K]

        # 3) восстанавливаем уровень и индекс внутри уровня
        #    offsets: [o0, o1, ..., o_{L-1}], где o_l = sum_{j<l} N_j
        offsets = torch.tensor(level_offsets, device=scores_all.device, dtype=inds_k.dtype)  # [L]
        # torch.bucketize вернёт l такое, что offsets[l] <= idx < offsets[l+1]
        # для этого нужны верхние границы; построим их:
        sizes = [s.numel() for s in level_scores]
        bounds = torch.tensor([sum(sizes[:i+1]) for i in range(L)], device=inds_k.device, dtype=inds_k.dtype)  # [L]
        lvl_ids = torch.bucketize(inds_k, bounds)  # [K], значения 0..L-1

        start_per_lvl = torch.cat([offsets, offsets.new_tensor([bounds[-1].item()])])  # [L+1], последняя — N
        start_for_pick = start_per_lvl[lvl_ids]  # [K]
        local_idx = inds_k - start_for_pick     # [K] индекс внутри уровня

        # 4) по уровням векторно собираем боксы/классы
        boxes_px_all = []
        labels_all   = []
        for l in range(L):
            mask_l = (lvl_ids == l)
            if not mask_l.any():
                continue

            C, H, W = level_shapes[l]
            idx_l = local_idx[mask_l]    # [K_l] в диапазоне [0, C*H*W)

            cls_l = (idx_l // (H * W)).to(torch.int64)  # [K_l]
            pix   =  idx_l % (H * W)                    # [K_l]
            xs = (pix % W).float()
            ys = (pix // W).float()

            # вытягиваем size/offset из карты уровня для этой картинки
            size_map   = size_outputs[l][b].permute(1, 2, 0).reshape(-1, 2)    # [H*W,2]
            offset_map = offset_outputs[l][b].permute(1, 2, 0).reshape(-1, 2)  # [H*W,2]

            so_idx = pix.long()  # [K_l]
            sizes   = size_map.index_select(0, so_idx)      # [K_l,2]
            offsets = offset_map.index_select(0, so_idx)    # [K_l,2]

            w = sizes[:, 0].relu()
            h = sizes[:, 1].relu()
            cx = xs + offsets[:, 0]
            cy = ys + offsets[:, 1]

            x1 = cx - 0.5 * w;  y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w;  y2 = cy + 0.5 * h
            boxes = torch.stack([x1, y1, x2, y2], dim=-1)   # [K_l,4]  — в координатах уровня

            # умножаем на stride ЭТОГО уровня → пиксели
            s = strides[l]
            boxes_px = boxes * s

            boxes_px_all.append(boxes_px)
            labels_all.append(cls_l)

        if len(boxes_px_all) == 0:
            batch_results.append((
                torch.empty((0, 4), device=cls_outputs[0].device),
                torch.empty((0,),  device=cls_outputs[0].device),
                torch.empty((0,),  device=cls_outputs[0].device, dtype=torch.long),
            ))
            continue

        boxes_px_all = torch.cat(boxes_px_all, dim=0)  # [K,4] (порядок по уровням, но это те же K)
        labels_all   = torch.cat(labels_all,   dim=0)  # [K]
        scores_sel   = scores_k                      # [K] — уже от top-K

        # 5) единый threshold и (опционально) NMS
        keep = scores_sel > score_thresh
        boxes_px_all = boxes_px_all[keep]
        scores_sel   = scores_sel[keep]
        labels_all   = labels_all[keep]

        if nms_iou is not None and boxes_px_all.numel() > 0:
            keep_idx = batched_nms(boxes_px_all, scores_sel, labels_all, iou_threshold=nms_iou)
            boxes_px_all = boxes_px_all[keep_idx]
            scores_sel   = scores_sel[keep_idx]
            labels_all   = labels_all[keep_idx]

        # клип по границам
        H, W = image_size
        boxes_px_all[:, 0].clamp_(0, W); boxes_px_all[:, 2].clamp_(0, W)
        boxes_px_all[:, 1].clamp_(0, H); boxes_px_all[:, 3].clamp_(0, H)

        batch_results.append((boxes_px_all, scores_sel, labels_all))

    return batch_results
