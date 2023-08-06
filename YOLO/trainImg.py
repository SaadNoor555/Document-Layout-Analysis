import os
from ultralytics import YOLO
import wandb

wandb.init(mode="disabled")

file_content = """
path: /media/abhijit/New Volume/dls2/code/datasets/badlad/badlad  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/train  # val images (relative to 'path') 128 images
test:  images/test

# Classes
names:
  0: image

"""

with open("yolov8.yaml", mode="w") as f:
    f.write(file_content)


model = YOLO("yolov8m-seg.pt")
model.train(data="yolov8.yaml", epochs=100, device=[0], imgsz=672, \
            overlap_mask=False, mask_ratio=2, save_period=50, patience=30)