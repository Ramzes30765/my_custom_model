import cv2
import numpy as np
import torch
from torchvision.utils import make_grid

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

def visualize_batch_detections(images, preds_list):
    """
    images: list of original images (np.ndarray или torch.Tensor [3,H,W])
    preds_list: list of (boxes, scores, labels) from model.predict()
    return: single grid image in CHW format (float, 0–1)
    """
    vis_images = []
    for img, preds in zip(images, preds_list):
        boxes, scores, labels = preds
        vis = visualize_boxes(img, boxes, scores, labels,)
        vis_tensor = torch.from_numpy(vis).permute(2, 0, 1).float() / 255.0
        vis_images.append(vis_tensor)

    grid = make_grid(vis_images, nrow=4, normalize=False)
    return grid  # Tensor [3, H, W]