import os
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from clearml import Task

from lightning_modules.coco_data_module import COCODetectionDataModule
from lightning_modules.lightning_model import SOTALitModule
from models.build_model import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Anchor-Free Detection Model")

    # основные параметры
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, nargs=2, default=[512, 512])
    parser.add_argument("--num_classes", type=int, default=80)
    parser.add_argument("--max_epochs", type=int, default=100)

    # пути к COCO
    parser.add_argument("--coco_root", type=str, required=True, help="Путь к каталогу COCO (с train2017, val2017 и annotations)")
    parser.add_argument("--train_ann", type=str, default="annotations/instances_train2017.json")
    parser.add_argument("--val_ann", type=str, default="annotations/instances_val2017.json")

    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(42)

    task = Task.init(project_name="Anchor-Free Detection", task_name="Train SOTA Model")
    task.connect(vars(args))

    dm = COCODetectionDataModule(
        root_dir=args.coco_root,
        ann_train=args.train_ann,
        ann_val=args.val_ann,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=os.cpu_count()
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
        save_top_k=1,
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
        precision=16,
        log_every_n_steps=10
    )

    trainer.fit(lit_model, datamodule=dm)


if __name__ == "__main__":
    main()
