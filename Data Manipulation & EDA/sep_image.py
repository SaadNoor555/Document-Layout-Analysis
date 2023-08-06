import cv2
import os
from tqdm import tqdm
import random

ORIG_IMG_DIR = 'datasets/badlad/badlad/images/train'
ORIG_LBL_DIR = 'datasets/badlad/badlad/labels/train'

def check_image(filename):
    '''returns true if the image referred by filename label has minority class (table) in it'''
    f = open(filename, 'r')
    lines = f.readlines()
    for line in lines:
        if line[0]=='2':
            return True    
    return False

def transform_ann(name):
    filename = os.path.join(ORIG_LBL_DIR, name)
    f = open(filename, 'r')
    lines = f.readlines()
    il = []
    for line in lines:
        if line[0]=='2':
            nl = '0'+line[1:]
            il.append(nl)
    wf = open(filename, 'w')
    for line in il:
        wf.write(line+'\n')


imgs = os.listdir(ORIG_IMG_DIR)
for img in tqdm(imgs):
    lbl = img[:-4]+'.txt'
    lbl_path = os.path.join(ORIG_LBL_DIR, lbl)
    img_path = os.path.join(ORIG_IMG_DIR, img)
    if check_image(lbl_path):
        transform_ann(lbl)
    
    else:
        r = random.randint(0, 99)
        if r<20:
            transform_ann(lbl)
        else:
            os.remove(lbl_path)
            os.remove(os.path.join(ORIG_IMG_DIR, img))