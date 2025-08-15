import os
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from clearml import Task
import matplotlib
matplotlib.use('Agg')

from lightning_modules.coco_data_module import COCODetectionDataModule
from lightning_modules.lightning_model import SOTALitModule
from models.build_model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Anchor-Free Detection Model")

    # основные параметры
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, nargs=2, default=[224, 224])
    parser.add_argument("--num_classes", type=int, default=80)
    parser.add_argument("--max_epochs", type=int, default=10)

    # пути к COCO
    parser.add_argument("--coco_root", type=str, required=True, help="Путь к каталогу COCO (с train2017, val2017 и annotations)", default='coco')
    parser.add_argument("--train_ann", type=str, default="annotations/instances_train2017.json")
    parser.add_argument("--val_ann", type=str, default="annotations/instances_val2017.json")

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)
    
    assert os.path.exists(os.path.join(args.coco_root, args.train_ann)), "train_ann not found"
    ann_path = os.path.join(args.coco_root, args.train_ann)
    print(f"Checking annotation path: {ann_path}")
    print("Exists:", os.path.exists(ann_path))

    task = Task.init(project_name="Anchor-Free Detection", task_name="Train SOTA Model")
    task.connect(vars(args))

    dm = COCODetectionDataModule(
        root_dir=args.coco_root,
        ann_train=args.train_ann,
        ann_val=args.val_ann,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=4
    )
    dm.setup()

    model = build_model(num_classes=args.num_classes)
    lit_model = SOTALitModule(
        model=model,
        learning_rate=args.lr,
        num_classes=args.num_classes,
        image_size=tuple(args.image_size)
    )

    logger = TensorBoardLogger("tb_logs", name="sota_model")
    csv_logger = CSVLogger("logs", name="sota_model")

    checkpoint_cb = ModelCheckpoint(
        monitor="val_total",
        save_top_k=3,
        mode="min",
        filename="best-{epoch:02d}-{val_total:.4f}",
        save_last=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=args.max_epochs,
        logger=[logger, csv_logger],
        callbacks=[checkpoint_cb, lr_monitor],
        precision="16-mixed",
        log_every_n_steps=10,
        val_check_interval=100,
        limit_val_batches=0.2,
        # check_val_every_n_epoch=1
    )

    trainer.fit(lit_model, datamodule=dm)


if __name__ == "__main__":
    main()
