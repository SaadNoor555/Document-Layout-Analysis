import os
from rotate import change_label, rotate_image
import cv2
import random
from tqdm import tqdm

TRAIN_IMG_DIR = 'datasets/badlad/badlad/images/train'
TRAIN_LABEL_DIR = 'datasets/badlad/badlad/labels/train'


def check_minority(filename):
    '''returns true if the image referred by filename label has minority class (table) in it'''
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        if line[0]=='3':
            return True    
    return False

imgList = os.listdir(TRAIN_IMG_DIR)
labList = os.listdir(TRAIN_LABEL_DIR)

angles = [90, 180, 270]

REVERSE_FLAG = False

for imgname in tqdm(imgList):
    labname = imgname[:-4]+'.txt'
    if check_minority(os.path.join(TRAIN_LABEL_DIR, labname)):
        src = cv2.imread(os.path.join(TRAIN_IMG_DIR, imgname))
        h, w, c = src.shape
        for angle in angles:       
            rotate_image(os.path.join(TRAIN_IMG_DIR, imgname), angle)
            change_label(os.path.join(TRAIN_LABEL_DIR, labname), angle, w, h)

    else:
        r = random.randint(0, 9)
        if r<2:
            angle = angles[random.randint(0, 2)]
            src = cv2.imread(os.path.join(TRAIN_IMG_DIR, imgname))
            h, w, c = src.shape
            rotate_image(os.path.join(TRAIN_IMG_DIR, imgname), angle)
            change_label(os.path.join(TRAIN_LABEL_DIR, labname), angle, w, h)