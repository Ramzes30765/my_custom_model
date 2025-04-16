import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
import torch


class CocoDetectionWrapper(CocoDetection):
    def __init__(self, img_folder, ann_file, transform=None):
        super().__init__(img_folder, ann_file)
        self.transform = transform

        # СОЗДАЁМ отображение category_id → индекс [0..N-1]
        coco_cat_ids = self.coco.getCatIds()
        self.cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(coco_cat_ids)}

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        boxes = []
        labels = []
        for obj in target:
            x, y, w, h = obj['bbox']
            boxes.append([x, y, x + w, y + h])
            # Преобразуем category_id → 0-based index
            labels.append(self.cat_id_to_index[obj['category_id']])

        boxes = np.array(boxes)
        labels = np.array(labels)

        if self.transform:
            transformed = self.transform(image=np.array(img), bboxes=boxes, class_labels=labels)
            img = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['class_labels'], dtype=torch.long)

        return img, {'boxes': boxes, 'labels': labels}


class COCODetectionDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, ann_train, ann_val, image_size=(512, 512), batch_size=8, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.ann_train = ann_train
        self.ann_val = ann_val
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        
        train_img_dir = os.path.join(self.root_dir, "train2017")
        val_img_dir = os.path.join(self.root_dir, "val2017")
        train_ann_path = os.path.join(self.root_dir, self.ann_train)
        val_ann_path = os.path.join(self.root_dir, self.ann_val)

        self.train_dataset = CocoDetectionWrapper(
            img_folder=train_img_dir,
            ann_file=train_ann_path,
            transform=self.train_transform()
        )
        self.val_dataset = CocoDetectionWrapper(
            img_folder=val_img_dir,
            ann_file=val_ann_path,
            transform=self.val_transform()
        )

    def train_transform(self):
        return A.Compose([
            A.Resize(*self.image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def val_transform(self):
        return A.Compose([
            A.Resize(*self.image_size),
            A.Normalize(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        images = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return images, targets
