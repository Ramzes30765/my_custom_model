import torch
import cv2
import numpy as np
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

def draw_boxes(img, boxes, labels=None, color=(0, 255, 0)):
    img = img.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if labels is not None:
            cv2.putText(img, str(labels[i].item()), (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img

def tensor_to_heatmap_img(tensor, orig_size):
    heatmap = tensor.detach().cpu().float().numpy()
    heatmap = np.clip(heatmap, 0, 1)
    heatmap = cv2.resize(heatmap, (orig_size[1], orig_size[0]))
    heatmap = np.uint8(heatmap * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap

def draw_mask_points(image, mask, color=(0, 255, 0)):
    mask = mask.squeeze().detach().cpu().numpy()
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    ys, xs = np.where(mask > 0.5)
    for x, y in zip(xs, ys):
        cv2.circle(image, (x, y), 2, color, -1)
    return image

def visualize_debug_targets(image, boxes, labels, heatmaps, masks, preds=None):
    """
    image: np.ndarray (H, W, 3)
    boxes: Tensor [N, 4]
    labels: Tensor [N]
    heatmaps: List[Tensor [C, H, W]]
    masks: List[Tensor [1, H, W]]
    preds: Optional[List[Tensor [C, H, W]]]
    """
    vis_images = []

    # 1. Оригинал
    img_raw = image.copy()
    vis_images.append(torch.from_numpy(img_raw).permute(2, 0, 1).float() / 255.0)

    # 2. С боксами
    img_boxes = draw_boxes(img_raw, boxes, labels)
    vis_images.append(torch.from_numpy(img_boxes).permute(2, 0, 1).float() / 255.0)

    for i, (heatmap, mask) in enumerate(zip(heatmaps, masks)):
        # Сумма по классам → heatmap
        heatmap_sum = heatmap.sum(dim=0)
        heatmap_img = tensor_to_heatmap_img(heatmap_sum, img_raw.shape[:2])
        heatmap_overlay = cv2.addWeighted(img_raw, 0.6, heatmap_img, 0.4, 0)

        # С центрами из маски
        heatmap_with_mask = draw_mask_points(heatmap_overlay, mask)
        vis_images.append(torch.from_numpy(heatmap_with_mask).permute(2, 0, 1).float() / 255.0)

        # Предсказания (опционально)
        if preds:
            pred_map = preds[i].sigmoid().sum(dim=0)
            pred_img = tensor_to_heatmap_img(pred_map, img_raw.shape[:2])
            pred_overlay = cv2.addWeighted(img_raw, 0.6, pred_img, 0.4, 0)
            vis_images.append(torch.from_numpy(pred_overlay).permute(2, 0, 1).float() / 255.0)

    target_size = (512, 512)  # любой удобный тебе размер
    vis_images_resized = [TF.resize(img, target_size) for img in vis_images]
    grid = make_grid(vis_images_resized, nrow=2)
    
    grid = make_grid(vis_images, nrow=2)
    return grid  # [3, H, W] тензор