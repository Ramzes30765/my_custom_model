import torch
import cv2
import numpy as np


# ------------------- Top-K Decoder -------------------
def decode_predictions(cls_outputs, size_outputs, offset_outputs, center_outputs, topk=100):
    results = []
    for cls_map, size_map, offset_map, center_map in zip(cls_outputs, size_outputs, offset_outputs, center_outputs):
        B, C, H, W = cls_map.shape
        cls_map = torch.sigmoid(cls_map)
        center_map = torch.sigmoid(center_map)
        cls_map = cls_map * center_map
        cls_map = cls_map.view(B, -1)
        scores, inds = torch.topk(cls_map, k=topk // 5) # ЭТО ЛЮТЫЙ КОСТЫЛЬ!!!!!!!!!

        classes = inds // (H * W)
        pixel_inds = inds % (H * W)

        xs = (pixel_inds % W).float()
        ys = (pixel_inds // W).float()

        size_map = size_map.permute(0, 2, 3, 1).reshape(B, -1, 2)
        offset_map = offset_map.permute(0, 2, 3, 1).reshape(B, -1, 2)

        sizes = torch.gather(size_map, 1, pixel_inds.unsqueeze(-1).repeat(1, 1, 2))
        offsets = torch.gather(offset_map, 1, pixel_inds.unsqueeze(-1).repeat(1, 1, 2))

        cx = xs + offsets[..., 0]
        cy = ys + offsets[..., 1]
        w, h = sizes[..., 0], sizes[..., 1]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        results.append((boxes, scores, classes))

    return results

# ------------------- Stride Estimator -------------------
def get_strides_from_feature_maps(input_image_shape, feature_maps):
    input_h, input_w = input_image_shape
    strides = []
    for fmap in feature_maps:
        _, _, h_i, w_i = fmap.shape
        stride_h = input_h // h_i
        stride_w = input_w // w_i
        assert stride_h == stride_w, "Non-uniform stride!"
        strides.append(stride_h)
    return strides

# ------------------- Post-processing -------------------
def postprocess_predictions(results, features, image_size, score_thresh):
    
    strides = get_strides_from_feature_maps(image_size, features)
    B = results[0][0].shape[0]
    batch_results = []
    for b in range(B):
        boxes_all, scores_all, classes_all = [], [], []
        for (boxes, scores, classes), stride in zip(results, strides):
            boxes_b = boxes[b] * stride
            scores_b = scores[b]
            classes_b = classes[b]

            # Apply score threshold here
            keep = scores_b > score_thresh
            boxes_b = boxes_b[keep]
            scores_b = scores_b[keep]
            classes_b = classes_b[keep]

            boxes_all.append(boxes_b)
            scores_all.append(scores_b)
            classes_all.append(classes_b)

        boxes_all = torch.cat(boxes_all, dim=0)
        scores_all = torch.cat(scores_all, dim=0)
        classes_all = torch.cat(classes_all, dim=0)

        H, W = image_size
        boxes_all[..., 0].clamp_(0, W)
        boxes_all[..., 1].clamp_(0, H)
        boxes_all[..., 2].clamp_(0, W)
        boxes_all[..., 3].clamp_(0, H)

        batch_results.append((boxes_all, scores_all, classes_all))

    return batch_results
