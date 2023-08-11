# Data Manipulation & EDA

Data Manipulation Scripts can be devided into three categories:

* Augmentation and Conversion
* Checking Integrity
* EDA

All of them along with the scripts representing them are listed here.

## Augmentaion and Conversion

Files that can be categorized as augmentation files are described below:

##### binarization.py

This script converts a folder of images to binarized image and moves them to destination folder. 

In order to run, set `img_dirTrain` and `img_dirTrainDest` to appropriate values and in terminal, run:

```
$ python binarization.py
```

##### coco_to_yolo.py

This script converts coco format annotation to yolo format annotation. This can be used to get yolo format annotation for publynet dataset (on any fold). It just converts the annotation for a specific fold that you downloaded, saving time and space.

In order to run, set  `ANN_DIR` as your coco format annotation path, `IMG_DIR` as your image directory, and `LBL_DIR` as your desired YOLO format label path and in terminal run:

```
$ python coco_to_yolo.py
```

##### createAugVal.py

This script applies rotation augmentation, with oversampling for images with table instance. Oversampling for table class is achieved by rotating images with table instances in three different angles (90, 180, 270 degrees) using rotate functions from rotate.py, and saved as seperate images. 20% of the images without tables are also rotated for augmentation.

In order to run, set `TRAIN_LBL_DIR` and `TRAIN_IMG_DIR` as original dataset label and image directory. Then in terminal run:

```
$ python createAugVal.py
```

##### filterPubTables.py

This script filters images containing tables from publynet dataset. 

In order to run, set  `ANN_DIR` as your coco format annotation path, `IMG_DIR` as your image directory, and `LBL_DIR` as your desired YOLO format label path and in terminal run:

```
$ python filterPubTables.py
```

##### histogram.py

This script converts all images to histogram equalized version of themselves and moves them to destination folder.

In order to run, set `img_dirTrain` and `img_dirTrainDest` to appropriate values and in terminal, run:

```
$ python histogram.py
```

##### rotate.py

This script provides two functions, one to rotate an image and save it and another to rotate the annotation for that image. both functions, These functions are used by other scripts for augmentation.

##### sep_image.py

This script seperates all the document images with image class in it with some extra files without image class and transforms annotation file to just contain image annotation. It takes img directory and annotation directory and it is used to make data for training a model to segment images in a document.

In order to run, set `ORIG_IMG_DIR` and `ORIG_IMG_DIR` to appropriate values and in terminal, run:

```
$ python sep_image.py
```

##### sep_publy_val.py

This scripts 20% image for each class to a given validation folder.
In order to run, set `LBL_DIR` and `IMG_DIR` as original dataset label and image directory and `LBL_VAL` and `IMG_DIR` as validation set label and image directory. Then in terminal run:

```
python sep_publy_val.py
```

##### shrink_mask.py

This script shrinks training label's textbox annotation, to train a model which will not overestimate textboxes during inference.

In order to run, set `IMG_DIR` and `LBL_DIR` to dataset's image and label dir and in terminal, run:

```
python shrink_mask.py
```


## Checking Integrity

Files that can be catagorized as checking and fixing integrity of dataset, (corruptions in dataset can cause during dataset manipulation) are described below:

##### checkDataset.py

This script checks if all images in dataset (train and validation) have their corresponding annotation file in labels folder.

In order to run, use appropriate values for dataset train and validation path variables and run using:

```
python checkDataset.py
```

##### delete_rot.py

After applying rotation augmentation on dataset, if you want to revert the dataset to it's original state, use this script to delete the augmented files.

In order to, set appropriate dataset image and label folder path values in variable and run

```
python delete_rot.py
```

##### fix_yolo_lbl

Due to some bugs, some scripts that work with labels might leave the label file with extra blank lines. Use this script to fix those label files.

##### move_publy.py

If you mix publy with badlad for training a specialized dataset for a particular class (i.e. table). Use this script to move the publynet files back to their original location


## EDA

This category only contains one file, it is described below:

##### visualize_ann.py

This script visualizes yolo format annotation by overlaying it on top of image. it is useful to check if the label augmentations are working properly.
