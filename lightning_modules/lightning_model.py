import random

import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from clearml import Task

from loss.compute_loss import TotalLoss
from utils.preprocess import generate_target_maps
from utils.postprocess import get_strides_from_feature_maps
from metrics.detection_metrics import DetectionMetricsWrapper
from metrics.compute_metrics import compute_det_metrics
from utils.vizualization import visualize_batch_detections
from utils.debug_viz import visualize_debug_targets_matplotlib


class SOTALitModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, num_classes=80, image_size=(512, 512)):
        super().__init__()
        self.model = model
        self.criterion = TotalLoss()
        self.lr = learning_rate
        self.num_classes = num_classes
        self.image_size = image_size  # (H, W)
        self.save_hyperparameters(ignore=['model'])

        # Метрики (torchmetrics)
        self.metrics = DetectionMetricsWrapper()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images)  # [B, 3, H, W]

        preds = self.model(images)
        gt_targets = self._build_targets(targets, preds["features"])
        
        loss_dict = self.criterion(
            preds,
            gt_targets,
            # debug=(batch_idx==0)
            )

        self.log_dict(
            {f"train_{k}": v for k, v in loss_dict.items()},
            on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0)
        )
        return loss_dict["total"]

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images)

        preds = self.model(images)
        gt_targets = self._build_targets(targets, preds["features"]) #[ ]
        loss_dict = self.criterion(
            preds,
            gt_targets,
            # debug=(batch_idx==0)
            )

        self.log_dict(
            {f"val_{k}": v for k, v in loss_dict.items()},
            on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0)
        )

        # Пропускаем sanity check
        if self.trainer.sanity_checking:
            return loss_dict["total"]

        # mAP + визуализация
        with torch.no_grad():
            for i in range(len(images)):
                img = images[i].unsqueeze(0)
                target = targets[i]
                pred = self.model.predict(img, image_size=self.image_size)[0]
                boxes, scores, labels = pred

                # Обновляем метрики
                compute_det_metrics(preds=(boxes, scores, labels), targets=target, metric_obj=self.metrics)
                
                batch_size = self.hparams.batch_size if hasattr(self.hparams, "batch_size") else 8
                if batch_idx == random.randint(0, batch_size - 1):
                    
                    vis_img = self.model.visualize(
                            image=images[i].detach().cpu(),
                            preds=(boxes, scores, labels),
                            class_names=None
                        )
                    vis_img = torch.from_numpy(vis_img).permute(2, 0, 1).float() / 255.0
                    self.logger.experiment.add_image(
                        tag=f"val/pred_epoch{self.current_epoch}_img{i}",
                        img_tensor=vis_img,
                        global_step=self.global_step
                    )
                    
                    orig_images = [t["orig_image"] for t in targets]
                    orig_sizes  = [t["original_size"] for t in targets]

                    preds_list = [
                        self.model.predict(img, image_size=self.image_size, original_size=size)
                        for img, size in zip(images, orig_sizes)
                    ]

                    fig = visualize_batch_detections(
                        images=orig_images,
                        preds_list=preds_list,
                        score_thresh=0.3,
                        class_names=None,
                        nrow=4
                    )

                    task = Task.current_task()
                    task.get_logger().report_matplotlib_figure(
                        title=f"val/random_detections/epoch_{self.current_epoch}",
                        series=f"step_{self.global_step}",
                        figure=fig,
                        iteration=self.global_step
                    )
                    plt.close(fig)

                return loss_dict["total"]


        return loss_dict["total"]

    def on_validation_epoch_end(self):
        
        if self.trainer.sanity_checking:
            return # не трогаем метрики, пока sanity check
        
        # Вычисляем и логируем метрики
        metrics = self.metrics.compute()
        self.log("val/map", metrics["map"], prog_bar=True)
        self.log("val/map_50", metrics["map_50"], prog_bar=True)
        self.log("val/map_75", metrics["map_75"], prog_bar=True)
        self.metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        T_max = self.hparams.max_epochs if hasattr(self.hparams, "max_epochs") else 10
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }

    def _build_targets(self, batch_targets, features):
        """
        Генерация таргетов под каждый уровень feature map.
        batch_targets: [{'boxes': Tensor[N, 4], 'labels': Tensor[N]}, ...]
        features: [P3, P4, P5, ...]
        """
        strides = get_strides_from_feature_maps(self.image_size, features)
        output_shapes = [(f.shape[2], f.shape[3]) for f in features]
        B = len(batch_targets)

        heatmaps, sizes, offsets = [], [], []

        for sample in batch_targets:
            boxes = sample["boxes"]
            labels = sample["labels"]

            per_level_heatmaps, per_level_sizes, per_level_offsets = [], [], []
            for (H, W), stride in zip(output_shapes, strides):
                heatmap, size, offset, _ = generate_target_maps(
                    boxes, labels,
                    output_shape=(H, W),
                    num_classes=self.num_classes,
                    stride=stride
                )
                per_level_heatmaps.append(heatmap)
                per_level_sizes.append(size)
                per_level_offsets.append(offset)

            heatmaps.append(per_level_heatmaps)
            sizes.append(per_level_sizes)
            offsets.append(per_level_offsets)

        return {
            "heatmap": [torch.stack([heatmaps[b][l] for b in range(B)]) for l in range(len(strides))],
            "size":    [torch.stack([sizes[b][l]    for b in range(B)]) for l in range(len(strides))],
            "offset":  [torch.stack([offsets[b][l]  for b in range(B)]) for l in range(len(strides))]
        }
