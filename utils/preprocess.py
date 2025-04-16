import torch
import torch.nn.functional as F

def generate_target_maps(boxes, classes, output_shape, num_classes, stride, sigma=2.0):
    """
    Генерирует target-карты для обучения модели обнаружения объектов.

    Аргументы:
      boxes: Tensor размером [N, 4] с ground truth bounding-box в формате [x1, y1, x2, y2]
      classes: Tensor размером [N] с индексами классов (0-indexed)
      output_shape: кортеж (H, W) – размер выходной карты (обычно размер карты признаков)
      num_classes: общее число классов
      stride: масштабный коэффициент, переводящий координаты исходного изображения в координаты карты признаков
      sigma: стандартное отклонение для построения гауссианы вокруг центра объекта

    Возвращает:
      heatmap: Tensor размером [num_classes, H, W]
      size_target: Tensor размером [2, H, W], где 0-й канал – ширина, 1-й – высота (относительно stride)
      offset_target: Tensor размером [2, H, W] – смещение центра объекта относительно целочисленной координаты на карте
      mask: Tensor размером [H, W] – бинарная маска (1, если в данной ячейке находится центр объекта)
    """
    device = boxes.device
    
    H, W = output_shape
    heatmap = torch.zeros((num_classes, H, W), dtype=torch.float32, device=device)
    size_target = torch.zeros((2, H, W), dtype=torch.float32, device=device)
    offset_target = torch.zeros((2, H, W), dtype=torch.float32, device=device)
    mask = torch.zeros((H, W), dtype=torch.float32, device=device)

    # Обработка каждого ground truth
    for i in range(boxes.shape[0]):
        # Извлекаем координаты и класс
        x1, y1, x2, y2 = boxes[i]
        cls = int(classes[i])
        # Вычисляем центр, ширину и высоту в координатах исходного изображения
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w_box = x2 - x1
        h_box = y2 - y1

        # Перевод центра в координаты карты признаков
        cx_f = cx / stride
        cy_f = cy / stride

        # Целочисленные координаты соответствующей клетки
        grid_x = int(cx_f)
        grid_y = int(cy_f)
        if grid_x < 0 or grid_x >= W or grid_y < 0 or grid_y >= H:
            continue  # Пропускаем объекты, попадающие вне области

        # Определяем окно для гауссианы – выбираем радиус в 3 sigma
        radius = int(3 * sigma)
        left = max(0, grid_x - radius)
        right = min(W, grid_x + radius + 1)
        top = max(0, grid_y - radius)
        bottom = min(H, grid_y + radius + 1)

        # Создаем локальную сетку для окна
        y_coords = torch.arange(top, bottom, dtype=torch.float32, device=device)
        x_coords = torch.arange(left, right, dtype=torch.float32, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        # Вычисляем квадрат расстояния до истинного центра (cx_f, cy_f)
        d2 = (xx - cx_f) ** 2 + (yy - cy_f) ** 2
        gauss_patch = torch.exp(-d2 / (2 * sigma * sigma))

        # Обновляем heatmap: в случае пересечений берем максимум
        heatmap[cls, top:bottom, left:right] = torch.max(
            heatmap[cls, top:bottom, left:right],
            gauss_patch
        )

        # Устанавливаем target для размеров в клетке-центре (размеры нормированные по stride)
        size_target[0, grid_y, grid_x] = w_box / stride
        size_target[1, grid_y, grid_x] = h_box / stride

        # Смещения – разница между действительным положением центра и ближайшей клеткой
        offset_target[0, grid_y, grid_x] = cx_f - grid_x
        offset_target[1, grid_y, grid_x] = cy_f - grid_y

        # Помечаем, что в данной клетке находится объект
        mask[grid_y, grid_x] = 1.0

    return heatmap, size_target, offset_target, mask