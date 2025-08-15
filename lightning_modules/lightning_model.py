import random

import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from clearml import Task

from loss.compute_loss import TotalLoss
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
        self.score_thresh = 0.1
        self.save_hyperparameters(ignore=['model'])

        # Метрики (torchmetrics)
        self.metrics = DetectionMetricsWrapper()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = torch.stack(images)  # [B, 3, H, W]

        preds = self.model(images)
        gt_targets = self.model.pred_head.build_gt(
            targets,
            preds["features"],
            self.num_classes,
            self.image_size,
            sigma=2.0
        )
        
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
        gt_targets = self.model.pred_head.build_gt(
            targets,
            preds["features"],
            self.num_classes,
            self.image_size,
            sigma=2.0
        )
        
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
            B = images.size(0)
            orig_images = [t.get("orig_image", None) for t in targets]
            orig_sizes  = [t["original_size"] for t in targets]
            orig_targets = [t["original_targets"] for t in targets]
            
            preds_list = []
            for i in range(B):
                boxes, scores, labels = self.model.predict(
                    images[i].unsqueeze(0),                     # [1,3,H,W]
                    image_size=self.image_size,
                    original_size=orig_sizes[i],
                    topk=100,
                    score_thresh=self.score_thresh,
                    nms_iou=None
                )
                preds_list.append((boxes, scores, labels))
                
            compute_det_metrics(
                preds=(boxes, scores, labels),
                targets=targets[i],
                metric_obj=self.metrics
            )
            
        fig = visualize_batch_detections(
            images=orig_images,
            preds_list=preds_list,
            score_thresh=self.score_thresh,
            class_names=None,
            nrow=4
        )
        
        self.logger.experiment.add_image("val/predictions_grid", fig, global_step=self.global_step)
        # task = Task.current_task()
        # task.get_logger().report_matplotlib_figure(
        #     title=f"val/random_detections/epoch_{self.current_epoch}",
        #     series=f"step_{self.global_step}",
        #     figure=fig,
        #     iteration=self.global_step
        # )
        plt.imsave('123.jpg', fig)
        plt.close(fig)
        
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