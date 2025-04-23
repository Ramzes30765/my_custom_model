import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import math
from matplotlib.patches import Rectangle

def visualize_boxes(image, boxes, scores, classes, class_names=None):

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.shape[0] == 3:  # [C, H, W] → [H, W, C]
            # Денормализация
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = image.clamp(0, 1)
            image = (image * 255).byte().permute(1, 2, 0).numpy()
        else:
            image = image.numpy()
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)

    image = image.copy()

    for box, score, cls in zip(boxes, scores, classes):

        x1, y1, x2, y2 = map(int, box.tolist())
        color = (0, 255, 0)
        label = f"{class_names[cls]}: {score:.2f}" if class_names else f"{int(cls)}: {score:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        ((text_width, text_height), _) = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image

def draw_boxes_on_ax(ax, image, boxes, scores, classes, class_names=None, score_thresh=0.3):
    """
    Рисует боксы и метки на указанном ax Matplotlib.

    Args:
        ax: matplotlib.axes.Axes
        image: np.ndarray HxWxC uint8 или float [0,1], или torch.Tensor CxHxW или HxWxC
        boxes: Tensor[N,4] или ndarray[N,4]
        scores: Tensor[N] или ndarray[N]
        classes: Tensor[N] или ndarray[N]
        class_names: список строк или None
        score_thresh: порог для отображения
    """
    # Конвертация tensor -> numpy HWC float [0,1]
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:  # CxHxW -> HxWxC
            img_np = np.transpose(img_np, (1, 2, 0))
        image = img_np
    # Если uint8, нормируем до [0,1]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = np.clip(image, 0.0, 1.0)

    ax.imshow(image)
    ax.axis('off')

    # Преобразуем тензоры
    boxes_np = boxes.detach().cpu().numpy() if isinstance(boxes, torch.Tensor) else np.array(boxes)
    scores_np = scores.detach().cpu().numpy() if isinstance(scores, torch.Tensor) else np.array(scores)
    classes_np = classes.detach().cpu().numpy() if isinstance(classes, torch.Tensor) else np.array(classes)

    # Фильтрация по порогу
    keep = scores_np > score_thresh
    boxes_np = boxes_np[keep]
    scores_np = scores_np[keep]
    classes_np = classes_np[keep]

    # Рисуем боксы и текст
    for box, score, cls in zip(boxes_np, scores_np, classes_np):
        x1, y1, x2, y2 = box.astype(int)
        width, height = x2 - x1, y2 - y1
        rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
        ax.add_patch(rect)
        label = f"{class_names[int(cls)] if class_names else int(cls)}: {score:.2f}"
        ax.text(x1, y1 - 4, label, fontsize=8, color='white',
                bbox=dict(facecolor='g', alpha=0.6, pad=0))


def visualize_batch_detections(images, preds_list, score_thresh=0.3, class_names=None, nrow=4):
    """
    Создает Figure с сеткой изображений и детекций через Matplotlib.

    Args:
        images: список np.ndarray или torch.Tensor
        preds_list: список кортежей (boxes, scores, classes)
        score_thresh: порог для отображения
        class_names: список имен классов или None
        nrow: число изображений в строке
    Returns:
        matplotlib.figure.Figure
    """
    num_imgs = len(images)
    if num_imgs == 0:
        return plt.figure()

    ncol = math.ceil(num_imgs / nrow)
    fig, axes = plt.subplots(ncol, nrow, figsize=(nrow * 4, ncol * 4))
    axes = np.array(axes).reshape(-1)

    for idx, (img, (boxes, scores, classes)) in enumerate(zip(images, preds_list)):
        ax = axes[idx]
        draw_boxes_on_ax(ax, img, boxes, scores, classes, class_names, score_thresh)

    # Скрыть лишние оси
    for ax in axes[num_imgs:]:
        ax.axis('off')

    plt.tight_layout()
    return fig