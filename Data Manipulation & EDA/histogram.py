import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2

img_dirTrain="datasets/badlad/badlad/images/train";
# img_dirVal= "datasets/badlad/heq/images/val";

img_dirTrainDest="datasets/badlad/badlad/images/train"
# img_dirValDest= "datasets/badlad/heq/images/val"

listTrain= os.listdir(img_dirTrain);
# listVal= os.listdir(img_dirVal);


def heq(input,src,dest):
    img = cv2.imread(os.path.join(src,input), cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file could not be read, check with os.path.exists()"

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    cv2.imwrite(os.path.join(dest,input),cl1)

from tqdm import tqdm
# if os.path.exists("datasets/badlad/heq")== False:
#     os.mkdir("datasets/badlad/heq");
# if os.path.exists("datasets/badlad/heq/images/")== False:
#     os.mkdir("datasets/badlad/heq/images/");
# if os.path.exists("datasets/badlad/heq/images/train")== False:
#     os.mkdir("datasets/badlad/heq/images/train");
# if os.path.exists("datasets/badlad/heq/images/val")== False:
#     os.mkdir("datasets/badlad/heq/images/val");

for imgName in tqdm(listTrain):
    heq(imgName,img_dirTrain,img_dirTrainDest)
   
# for imgName in tqdm(listVal):
#     heq(imgName,img_dirVal,img_dirValDest)
   

