#!/bin/bash

# Папка, куда всё будет скачано
COCO_DIR="./coco"

mkdir -p "$COCO_DIR"
cd "$COCO_DIR"

echo "Скачиваем COCO изображения и аннотации..."

# Изображения
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip

# Аннотации
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

echo "Распаковываем..."

unzip -q train2017.zip
unzip -q val2017.zip
unzip -q annotations_trainval2017.zip

echo "Удаляем .zip файлы..."
rm train2017.zip val2017.zip annotations_trainval2017.zip

echo "✅ Готово! COCO находится в: $PWD"
