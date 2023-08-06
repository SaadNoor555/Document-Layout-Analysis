import os
from tqdm import tqdm

img_dir = 'datasets/badlad/badlad/images/train'
lbl_dir = 'datasets/badlad/badlad/labels/train'

for imgname in tqdm(os.listdir(img_dir)):
    if '_90' in imgname or '_180' in imgname or '_270' in imgname:
        os.remove(os.path.join(img_dir, imgname))
        os.remove(os.path.join(lbl_dir, imgname[:-4]+'.txt'))