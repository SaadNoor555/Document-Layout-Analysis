import os
from ultralytics import YOLO
import wandb
import torch, gc
gc.collect()
torch.cuda.empty_cache()
wandb.init(mode="disabled")

file_content = """
task: segment
path: /media/abhijit/New Volume/dls2/code/datasets/badlad/badlad  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/train  # val images (relative to 'path') 128 images
test: images/test

# Classes
names:
  0: paragraph
  1: text_box
  2: image
  3: table

"""

with open("yolov8_aug_2.yaml", mode="w") as f:
    f.write(file_content)


model = YOLO("yolov8m-seg.pt")
model.train(data="yolov8_aug_2.yaml", epochs=50, device=[0], imgsz=672, \
            overlap_mask=True, mask_ratio=2, save_period=10, augment=True,\
            copy_paste=0.1, mixup=0.1, mosaic=0.5, perspective=0.001, cos_lr=True, degrees=90
            )