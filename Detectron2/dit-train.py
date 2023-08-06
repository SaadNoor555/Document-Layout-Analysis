


import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


from datetime import datetime

# if False, model is set to `PRETRAINED_PATH` model
is_train = True

# if True, evaluate on validation dataset
is_evaluate = True

# if True, run inference on test dataset
is_inference = True

# if True and `is_train` == True, `PRETRAINED_PATH` model is trained further
is_resume_training = False

# Perform augmentation
is_augment = False

SEED = 42
import random
import os
import numpy as np
import torch
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

"""## 2.2 Paths"""

from pathlib import Path

TRAIN_IMG_DIR = Path("/media/abhijit/New Volume/dls2/code/datasets/badlad/badlad/images/train")

TRAIN_COCO_PATH = Path("/media/abhijit/New Volume/dls2/code/datasets/badlad/badlad/labels/coco_format/train/badlad-train-coco.json")

TEST_IMG_DIR = Path("/media/abhijit/New Volume/dls2/code/datasets/badlad/badlad/images/test")

TEST_METADATA_PATH = Path("/media/abhijit/New Volume/dls2/code/datasets/badlad/badlad/badlad-test-metadata.json")

# Training output directory
OUTPUT_DIR = Path("./output")
OUTPUT_MODEL = OUTPUT_DIR/"model_final.pth"

# Path to your pretrained model weights
PRETRAINED_PATH = Path("/media/abhijit/New Volume/dls2/code/publaynet_dit-b_cascade.pth")

# Model path based on Decisions
MODEL_PATH = OUTPUT_MODEL if is_train else PRETRAINED_PATH

"""## 2.3 imports"""

# detectron2
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo

import pandas as pd
import numpy as np
from tqdm.notebook import tqdm  # progress bar
import matplotlib.pyplot as plt
import json
import cv2
import copy
from typing import Optional

from IPython.display import FileLink

# torch
import torch
import os

import gc

import warnings
# Ignore "future" warnings and Data-Frame-Slicing warnings.
warnings.filterwarnings('ignore')

setup_logger()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


import sys
sys.path.append("unilm")
from unilm.dit.object_detection.ditod import add_vit_config

"""# 3 COCO Annotations Data

## 3.1 Load
"""

with TRAIN_COCO_PATH.open() as f:
    train_dict = json.load(f)

# with TEST_METADATA_PATH.open() as f:
#     test_dict = json.load(f)

print("#### LABELS AND METADATA LOADED ####")

"""## 3.2 Observe"""

def organize_coco_data(data_dict: dict):
    thing_classes: list[str] = []

    # Map Category Names to IDs
    for cat in data_dict['categories']:
        thing_classes.append(cat['name'])

    print(thing_classes)

    # thing_classes = ['paragraph', 'text_box', 'image', 'table']
    # Images
    images_metadata: list[dict] = data_dict['images']

    # Convert COCO annotations to detectron2 annotations format
    data_annotations = []
    for ann in data_dict['annotations']:
        # coco format -> detectron2 format
        annot_obj = {
            # Annotation ID
            "id": ann['id'],

            # Segmentation Polygon (x, y) coords
            "gt_masks": ann['segmentation'],

            # Image ID for this annotation (Which image does this annotation belong to?)
            "image_id": ann['image_id'],

            # Category Label (0: paragraph, 1: text box, 2: image, 3: table)
            "category_id": ann['category_id'],

            "x_min": ann['bbox'][0],  # left
            "y_min": ann['bbox'][1],  # top
            "x_max": ann['bbox'][0] + ann['bbox'][2],  # left+width
            "y_max": ann['bbox'][1] + ann['bbox'][3]  # top+height
        }
        data_annotations.append(annot_obj)

    return thing_classes, images_metadata, data_annotations

thing_classes, images_metadata_train, train_data_annotations = organize_coco_data(
    train_dict
)

# thing_classes_test, images_metadata_test, _ = organize_coco_data(
#     test_dict
# )

train_metadata = pd.DataFrame(images_metadata_train)
train_metadata = train_metadata[['id', 'file_name', 'width', 'height']]
train_metadata = train_metadata.rename(columns={"id": "image_id"})
print("train_metadata size=", len(train_metadata))
train_metadata.head(5)

train_annot_df = pd.DataFrame(train_data_annotations)
print("train_annot_df size=", len(train_annot_df))
train_annot_df.head(5)

"""Here `gt_masks` are the sequence of `(x, y)` coordinates of vertices of the polygon surrounding the target object."""

# test_metadata = pd.DataFrame(images_metadata_test)
# test_metadata = test_metadata[['id', 'file_name', 'width', 'height']]
# test_metadata = test_metadata.rename(columns={"id": "image_id"})
# print("test_metadata size=", len(test_metadata))
# test_metadata.head(5)

"""These are the categories we are going to detect."""

print(thing_classes)

DATA_REGISTER_TRAINING = "badlad_train"
DATA_REGISTER_VALID    = "badlad_valid"
DATA_REGISTER_TEST     = "badlad_test"

train_df_with_annotations = train_metadata.merge(train_annot_df, on='image_id')
train_df_with_annotations.head()

# Group annotations by image_id and aggregate into a list
train_df_with_annotations_grouped = train_df_with_annotations.groupby('image_id').agg(lambda x: x.tolist()).reset_index()
train_df_with_annotations_grouped.head()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()
gc.collect()

"""# 4 Preparing Data for Training

## 4.1 Train-Validation Split
"""

"""## 4.2 Formatting Data for `detectron2`"""

def convert_coco_to_detectron2_format(
    imgdir: Path,
    metadata_df: pd.DataFrame,
    annot_df: Optional[pd.DataFrame] = None,
    target_indices: Optional[np.ndarray] = None,
):

    dataset_dicts = []
    for _, train_meta_row in metadata_df.iterrows():
    # Your code for each row goes here

        # Iterate over each image
        image_id, filename, width, height = train_meta_row.values

        annotations = []

        # If train/validation data, then there will be annotations
        if annot_df is not None:
            for _, ann in annot_df.query("image_id == @image_id").iterrows():
                # Get annotations of current iteration's image
                class_id = ann["category_id"]
                gt_masks = ann["gt_masks"]
                bbox_resized = [
                    float(ann["x_min"]),
                    float(ann["y_min"]),
                    float(ann["x_max"]),
                    float(ann["y_max"]),
                ]

                annotation = {
                    "bbox": bbox_resized,
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": gt_masks,
                    "category_id": class_id,
                }

                annotations.append(annotation)

        # coco format -> detectron2 format dict
        record = {
            "file_name": str(imgdir/filename),
            "image_id": image_id,
            "width": width,
            "height": height,
            "annotations": annotations
        }

        dataset_dicts.append(record)

    if target_indices is not None:
        dataset_dicts = [dataset_dicts[i] for i in target_indices]

    return dataset_dicts

# Create empty lists to store images with tables and images without tables
images_with_tables = []
images_without_tables = []

# Loop through each row in the DataFrame
for _, row in train_df_with_annotations_grouped.iterrows():
    # Check if category_id for "table" (3) is present in the list
    if 3 in row['category_id']:
        images_with_tables.append(row['image_id'])
    else:
        images_without_tables.append(row['image_id'])

# Create DataFrames for images with tables and images without tables
images_with_tables_df = train_metadata[train_metadata['image_id'].isin(images_with_tables)]
images_without_tables_df = train_metadata[train_metadata['image_id'].isin(images_without_tables)]

# Convert the DataFrames to detectron2 format:
dataset_dicts_with_tables = convert_coco_to_detectron2_format(TRAIN_IMG_DIR, images_with_tables_df, train_annot_df)
dataset_dicts_without_tables = convert_coco_to_detectron2_format(TRAIN_IMG_DIR, images_without_tables_df, train_annot_df)

# Now, you have two separate datasets: dataset_dicts_with_tables containing images with tables annotated,
# and dataset_dicts_without_tables containing images without tables annotated.


# Create DataFrames for images with tables and images without tables
images_with_tables_df = train_metadata[train_metadata['image_id'].isin(images_with_tables)]
images_without_tables_df = train_metadata[train_metadata['image_id'].isin(images_without_tables)]


# Calculate the number of images with tables and without tables
num_images_with_tables = len(dataset_dicts_with_tables)
num_images_without_tables = len(dataset_dicts_without_tables)

from sklearn.model_selection import train_test_split
import random
# Perform a stratified split on the dataset with tables
dataset_with_tables_train, dataset_with_tables_valid = train_test_split(dataset_dicts_with_tables, test_size=0.2,)

# Perform a stratified split on the dataset without tables
dataset_without_tables_train, dataset_without_tables_valid = train_test_split(dataset_dicts_without_tables, test_size=0.2)

# Concatenate the datasets to create the final balanced training and validation sets
balanced_dataset_train = dataset_with_tables_train + dataset_without_tables_train
balanced_dataset_valid = dataset_with_tables_valid + dataset_without_tables_valid

# Shuffle the datasets to further randomize the order of images
random.shuffle(balanced_dataset_train)
random.shuffle(balanced_dataset_valid)

def check_category_statistics(dataset):
    category_counts = {}
    for data in dataset:
        annotations = data["annotations"]
        for annotation in annotations:
            category_id = annotation["category_id"]
            if category_id not in category_counts:
                category_counts[category_id] = 0
            category_counts[category_id] += 1

    category_statistics = {
        "category_id": [],
        "category_name": [],
        "num_instances": []
    }
    for category_id, num_instances in category_counts.items():
        category_statistics["category_id"].append(category_id)
        category_statistics["category_name"].append(thing_classes[category_id])
        category_statistics["num_instances"].append(num_instances)

    category_statistics_df = pd.DataFrame(category_statistics)
    return category_statistics_df

train_category_statistics = check_category_statistics(balanced_dataset_train)
valid_category_statistics = check_category_statistics(balanced_dataset_valid)

print("Training Set Category Statistics:")
print(train_category_statistics)

print("\nValidation Set Category Statistics:")
print(valid_category_statistics)



"""## 4.3 Registering and Loading Data for `detectron2`"""

DATA_REGISTER_TRAINING = "badlad_train"
DATA_REGISTER_VALID    = "badlad_valid"
DATA_REGISTER_TEST     = "badlad_test"

# Register Training data
if is_train:
    DatasetCatalog.register(DATA_REGISTER_TRAINING, lambda: balanced_dataset_train)
    MetadataCatalog.get(DATA_REGISTER_TRAINING).set(thing_classes=thing_classes)
    metadata_dicts_train = MetadataCatalog.get(DATA_REGISTER_TRAINING)
    print("dicts training size=", len(balanced_dataset_train))
    print("################")

# Register Validation data
if is_train or is_evaluate:
    DatasetCatalog.register(DATA_REGISTER_VALID, lambda: balanced_dataset_valid)
    MetadataCatalog.get(DATA_REGISTER_VALID).set(thing_classes=thing_classes)
    metadata_dicts_valid = MetadataCatalog.get(DATA_REGISTER_VALID)
    print("dicts valid size=", len(balanced_dataset_valid))
    print("################")

# Register Test Inference data
# DatasetCatalog.register(
#     DATA_REGISTER_TEST,
#     lambda: convert_coco_to_detectron2_format(
#         TEST_IMG_DIR,
#         test_metadata,
#     )
# )

# Set Test data categories
# MetadataCatalog.get(DATA_REGISTER_TEST).set(
#     thing_classes=thing_classes_test
# )

# dataset_dicts_test = DatasetCatalog.get(DATA_REGISTER_TEST)
metadata_dicts_test = MetadataCatalog.get(DATA_REGISTER_TEST)

print("#### DATA REGISTERED ####")

"""# 5 Augmentation"""

def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")

    transform_list = [
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        # T.RandomExtent(scale_range=(0.8, 1.2), shift_range=(-0.1, 0.1)),
        # T.RandomRotation(angle=(-10, 10))
    ]
    
    # Apply augmentations directly on the original image (in-place)
    for t in transform_list:
        image = t.apply_image(image)

    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])

    dataset_dict["instances"] = utils.filter_empty_instances(instances)

    return dataset_dict


class AugTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.cuda.empty_cache()
gc.collect()

cfg = get_cfg()
add_vit_config(cfg)
cfg.merge_from_file("/media/abhijit/New Volume/dls2/code/DIT-training/cascade_dit_base.yaml")
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.WEIGHTS = "/media/abhijit/New Volume/dls2/code/DIT-training/dit-pub-20000.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
cfg.SOLVER.BASE_LR = 0.0003
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.AMP.ENABLED = True
cfg.SOLVER.WARMUP_ITERS = 200
cfg.DATALOADER.NUM_WORKERS = 2
# cfg.SOLVER.STEPS = (2500, 5000, 8000, 10000, 12000, 14000, 16000, 18000, 20000, 21500, 23000, 24500, 26000, 27000, 28000, 29000, 30000, 31000, 32000, 33000, 34000, 35000)
# cfg.SOLVER.GAMMA = 0.7
cfg.SOLVER.MAX_ITER = 80000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.SOLVER.CHECKPOINT_PERIOD = 2500
cfg.DATASETS.TRAIN = (DATA_REGISTER_TRAINING,)
cfg.DATASETS.TEST = (DATA_REGISTER_VALID,)
cfg.TEST.EVAL_PERIOD = 80000


# Create Output Directory
cfg.OUTPUT_DIR = str(OUTPUT_DIR)
print("creating cfg.OUTPUT_DIR -> ", cfg.OUTPUT_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)


from detectron2.engine import (
    hooks,
)
from detectron2.config import instantiate


def do_test(cfg, model):
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.55   # set a custom testing threshold
    cfg.SOLVER.IMS_PER_BATCH = 64
    
    evaluator = COCOEvaluator(
        DATA_REGISTER_VALID, cfg, False, output_dir=cfg.OUTPUT_DIR, use_fast_impl=True
    )

    val_loader = build_detection_test_loader(cfg, DATA_REGISTER_VALID)

    results = inference_on_dataset(
        trainer.model, val_loader, evaluator=evaluator
    )


from unilm.dit.object_detection.ditod import MyTrainer

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
# trainer.register_hooks(
#         [
#             hooks.EvalHook(cfg.TEST.EVAL_PERIOD, lambda: do_test(cfg, trainer.model)),
#         ]
#     )
# Uncomment to train
trainer.train()


predictor = DefaultPredictor(cfg)

test_data = [{"file_name": "/media/abhijit/New Volume/dls2/code/datasets/badlad/badlad/images/test/0a0a76d0-2ee5-47f8-aa6e-afd93fd6c1a0.png"}]
im = cv2.imread(test_data[0]["file_name"])
outputs = predictor(im)
print(outputs)
v = Visualizer(im[:, :, ::-1],
               metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
               scale=0.5,
               instance_mode=ColorMode.IMAGE_BW)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
img = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_RGBA2RGB)
plt.imshow(img)
