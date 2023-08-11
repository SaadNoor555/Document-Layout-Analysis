# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2

# from skimage.data import page
# from skimage.filters import (threshold_otsu, threshold_niblack,
#                              threshold_sauvola)
# import argparse
import os
import cv2
import tqdm

img_dirTrain="datasets/badlad/badlad/images/train";
# img_dirVal= "datasets/badlad/badlad/images/val";

img_dirTrainDest="datasets/badlad/badlad/images/train"
# img_dirValDest= "datasets/badlad/badlad/images/val"

listTrain= os.listdir(img_dirTrain);
# listVal= os.listdir(img_dirVal);


def binarize(input,src,dest):
    img = cv2.imread(os.path.join(src,input), cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"
    window_size=25
    thresh_sauvola = threshold_sauvola(img, window_size=window_size)
    binary_sauvola = img > thresh_sauvola
    cv2.imwrite(os.path.join(dest,input),binary_sauvola)



def convert(fn):
    # given a file name, convert it into binary and store at the same position
    img = cv2.imread(fn)
    gim = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gim = cv2.adaptiveThreshold(gim, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 12.36)
    g3im = cv2.cvtColor(gim, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(fn, g3im)


from tqdm import tqdm
# if os.path.exists("datasets/badlad/bin")== False:
#     os.mkdir("datasets/badlad/bin");
# if os.path.exists("datasets/badlad/bin/images/")== False:
#     os.mkdir("datasets/badlad/bin/images/");
# if os.path.exists("datasets/badlad/bin/images/train")== False:
#     os.mkdir("datasets/badlad/bin/images/train");
# if os.path.exists("datasets/badlad/bin/images/val")== False:
#     os.mkdir("datasets/badlad/bin/images/val");

for imgName in tqdm(listTrain):
    convert(os.path.join(img_dirTrain,imgName))
   
# for imgName in tqdm(listVal):
#     convert(os.path.join(img_dirVal,imgName))
   
# python3 binarization.py
