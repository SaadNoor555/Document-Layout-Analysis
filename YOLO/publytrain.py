import os
from ultralytics import YOLO
import wandb
import torch, gc
gc.collect()
torch.cuda.empty_cache()

file_content = """
path: publynet  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images

# Classes
names:
  0: text
  1: title
  2: list
  3: table
  4: figure
"""

with open("publynet.yaml", mode="w") as f:
    f.write(file_content)
torch.cuda.empty_cache()

from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")
model.train(data="publynet.yaml", epochs=50, device=[0], imgsz=672, \
            overlap_mask=True, mask_ratio=2, save_period=25)