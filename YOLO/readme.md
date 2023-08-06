# YOLO

This folder contains all the scripts used to train yolo models.

#### publytrain.py

This script trains a YOLOv8 model on publynet dataset. edit the `file_content` to your meet your criteria and run `python publytrain.py` to start training. Before running this script, you have to download the publynet dataset, and it's labels, then use `Data Manipulation & EDA\coco_to_yolo.py` script to convert annotation to appropriate format and then start the training.

#### trainImg.py

This script trains a YOLOv8 model to recognize images in a document. In order to run this script, you have to first prepare your dataset by running `Data Manipulation & EDA\sep_image.py` script. After your dataset is ready, specify it's locations in `file_content` and run the script using `python trainimg.py` to start training.

#### yolo_raw.py

This script trains a YOLOv8 model on badlad dataset (original or histogram equalized/binarized). In order to start training, you have to download badlad dataset, move yolov8 train label to appropriate location, seperate validation set if you may, apply augmentations (optional) and run the script using python `yolo_raw.py`
