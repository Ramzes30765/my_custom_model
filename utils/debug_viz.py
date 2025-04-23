import torch
import cv2
import numpy as np
from torchvision.utils import make_grid

# def draw_boxes(img, boxes, labels=None, color=(0, 255, 0)):
#     """
#     img: np.ndarray H×W×3, uint8 RGB
#     boxes: Tensor[N,4]
#     labels: Tensor[N] или None
#     """
#     img = img.copy()
#     for i, box in enumerate(boxes):
#         x1, y1, x2, y2 = map(int, box.tolist())
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         if labels is not None:
#             cv2.putText(img, str(labels[i].item()), (x1, y1 - 2),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#     return img

# def tensor_to_heatmap_img(tensor, orig_size):
#     """
#     tensor: Torch [H_feat, W_feat], float in [0,1]
#     orig_size: (H, W)
#     Возвращает uint8 RGB H×W×3
#     """
#     heatmap = tensor.detach().cpu().numpy()
#     heatmap = np.clip(heatmap, 0.0, 1.0)
#     heatmap = cv2.resize(heatmap, (orig_size[1], orig_size[0]))
#     heatmap = (heatmap * 255).astype(np.uint8)               # CV_8U
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)   # BGR
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)       # → RGB
#     return heatmap

# def draw_mask_points(image, mask, color=(0, 255, 0)):
#     """
#     image: np.ndarray H×W×3 uint8 RGB
#     mask: Torch [1, H_feat, W_feat]
#     """
#     img = image.copy()
#     m = mask.squeeze().detach().cpu().numpy()
#     m = cv2.resize(m, (img.shape[1], img.shape[0]))
#     ys, xs = np.where(m > 0.5)
#     for x, y in zip(xs, ys):
#         cv2.circle(img, (x, y), 2, color, -1)
#     return img

# def visualize_debug_targets(image, boxes, labels, heatmaps, masks, preds=None):
#     """
#     Сравнение: оригинал, GT-боксы, GT-heatmap+mask и (опционально) pred-heatmap.

#     image: np.ndarray H×W×3 uint8 RGB
#     boxes: Tensor[N,4]
#     labels: Tensor[N]
#     heatmaps: List[Tensor[C, H_feat, W_feat]]
#     masks:    List[Tensor[1, H_feat, W_feat]]
#     preds:    None или List[Tensor[C, H_feat, W_feat]]
#     """
#     vis = []
#     H, W = image.shape[:2]

#     # 1) raw RGB
#     img_raw = image.copy()
#     vis.append(torch.from_numpy(img_raw)[None].permute(1, 2, 3, 0)[...,0].permute(2, 0, 1).float() / 255.0)

#     # 2) raw + GT boxes
#     img_boxes = draw_boxes(img_raw, boxes, labels)
#     vis.append(torch.from_numpy(img_boxes).permute(2, 0, 1).float() / 255.0)

#     # 3+) по уровням
#     for lvl, (hmap, msk) in enumerate(zip(heatmaps, masks)):
#         # GT heatmap
#         hm = hmap.sum(dim=0)
#         hm_img = tensor_to_heatmap_img(hm, (H, W))
#         overlay = cv2.addWeighted(img_raw, 0.6, hm_img, 0.4, 0)
#         overlay = draw_mask_points(overlay, msk)
#         vis.append(torch.from_numpy(overlay).permute(2, 0, 1).float() / 255.0)

#         # Pred heatmap, если задано
#         if preds is not None:
#             pm = preds[lvl].sigmoid().sum(dim=0)
#             pm_img = tensor_to_heatmap_img(pm, (H, W))
#             pred_overlay = cv2.addWeighted(img_raw, 0.6, pm_img, 0.4, 0)
#             vis.append(torch.from_numpy(pred_overlay).permute(2, 0, 1).float() / 255.0)

#     # Собираем все в сетку 2×N
#     grid = make_grid(vis, nrow=2)
#     return grid

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_debug_targets_matplotlib(
    image: np.ndarray,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    heatmaps: list,   # List[Tensor[C, Hf, Wf]]
    masks: list,      # List[Tensor[1, Hf, Wf]]
    preds: list = None  # Optional[List[Tensor[C, Hf, Wf]]]
):
    """
    1) Raw image
    2) Image + GT-боксы
    3) Для каждого уровня:
         - GT-heatmap (sum over classes) с alpha=0.4
         - белые точки из mask>0.5
         - (опц.) pred-heatmap
    """

    # 1) Подготовка входов
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    H, W = img.shape[:2]

    n_levels = len(heatmaps)
    # На каждый уровень 1 график GT, + еще 1 pred, если есть preds
    per_level = 1 + (1 if preds is not None else 0)
    n_plots = 2 + n_levels * per_level  # raw + boxes + уровни

    # Разбивка на строки/столбцы
    n_cols = int(np.ceil(n_plots / 2))
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten()

    idx = 0

    # ——— 1) Raw image
    ax = axes[idx]
    ax.imshow(img, origin='upper')
    ax.set_title("Raw Image")
    ax.axis('off')
    idx += 1

    # ——— 2) GT-боксы
    ax = axes[idx]
    ax.imshow(img, origin='upper')
    ax.set_title("GT Boxes")
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), w, h,
                         edgecolor='lime', facecolor='none', lw=2)
        ax.add_patch(rect)
        ax.text(x1, y1-3, str(int(labels[i].item())),
                color='lime', fontsize=8,
                backgroundcolor='black')
    ax.axis('off')
    idx += 1

    # ——— 3+) По каждому уровню
    for lvl, (hmap, msk) in enumerate(zip(heatmaps, masks)):
        # GT-heatmap
        hm = hmap.sum(dim=0).detach().cpu().numpy()
        hm = np.clip(hm, 0.0, 1.0)

        ax = axes[idx]
        ax.imshow(img, origin='upper')
        # extent=[xmin,xmax,ymin,ymax] растягивает карту на full image
        ax.imshow(hm,
                  cmap='jet',
                  alpha=0.4,
                  vmin=0, vmax=1,
                  interpolation='nearest',
                  extent=[0, W, H, 0],
                  origin='upper')
        ax.set_title(f"Level {lvl} GT Heatmap")
        ax.axis('off')

        # Центры из mask
        mm = msk.squeeze().detach().cpu().numpy()
        ys, xs = np.where(mm > 0.5)
        # Масштабируем координаты из (Hf,Wf) → (H,W)
        ys = ys * (H / mm.shape[0])
        xs = xs * (W / mm.shape[1])
        ax.scatter(xs, ys, s=8, c='white', marker='o')
        idx += 1

        # Pred-heatmap (если есть)
        if preds is not None:
            pm = preds[lvl].sigmoid().sum(dim=0).detach().cpu().numpy()
            pm = np.clip(pm, 0.0, 1.0)

            ax = axes[idx]
            ax.imshow(img, origin='upper')
            ax.imshow(pm,
                      cmap='jet',
                      alpha=0.4,
                      vmin=0, vmax=1,
                      interpolation='nearest',
                      extent=[0, W, H, 0],
                      origin='upper')
            ax.set_title(f"Level {lvl} Pred Heatmap")
            ax.axis('off')
            idx += 1

    # Убираем лишние оси, если они есть
    for j in range(idx, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    return fig

