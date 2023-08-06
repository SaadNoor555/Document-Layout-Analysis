import os
from tqdm import tqdm

TRAIN_IMG_DIR = 'datasets/badlad/heq/images/train'
TRAIN_LABEL_DIR = 'datasets/badlad/heq/labels/train'
VAL_IMG_DIR = 'datasets/badlad/heq/images/val'
VAL_LABEL_DIR = 'datasets/badlad/heq/labels/val'

no1, no2 = 0, 0

def check(img_dir, lbl_dir, no):
    for img in tqdm(os.listdir(img_dir)):
        lbl = img[:-4]+'.txt'
        try:
            imgfile = open(os.path.join(img_dir, img), 'r')
            lblfile = open(os.path.join(lbl_dir, lbl), 'r')
        except:
            no+=1

    return no

print(check(TRAIN_IMG_DIR, TRAIN_LABEL_DIR, no1))
print(check(VAL_IMG_DIR, VAL_LABEL_DIR, no2))