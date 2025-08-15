FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install \
    torch torchvision torchmetrics \
    pytorch-lightning \
    albumentations albucore \
    opencv-python \
    pycocotools \
    timm \
    clearml clearml-agent \
    matplotlib tensorboard \
    numpy scipy pandas \
    tqdm

RUN mkdir -p /root/coco && cd /root/coco && \
    wget -nc http://images.cocodataset.org/zips/train2017.zip && \
    wget -nc http://images.cocodataset.org/zips/val2017.zip && \
    wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip && \
    unzip -q -n train2017.zip && \
    unzip -q -n val2017.zip && \
    unzip -q -n annotations_trainval2017.zip && \
    rm -f train2017.zip val2017.zip annotations_trainval2017.zip

WORKDIR /root