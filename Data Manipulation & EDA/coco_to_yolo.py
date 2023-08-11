import json
import os
import cv2
from tqdm import tqdm

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

ANN_DIR = 'datasets/labels/publaynet/train.json'
IMG_DIR = 'datasets/train-1/publaynet/train'
LBL_DIR = 'datasets/train-1/publaynet/labels/train'

print('-------loading annotation file--------')
f = open(ANN_DIR, 'r')
fj = json.load(f)
print('-------annotation file loaded--------')

cats = fj['categories']
imgs = fj['images']
anns = fj['annotations']
img_dict = {}
lbl_ann_dict = {}


print('---------getting image ids--------')
for img in tqdm(imgs):
    img_name = img['file_name']
    img_path = os.path.join(IMG_DIR, img_name)
    try:
        img_file = open(img_path)
        img_id = img['id']
        img_dict[img_id] = img_name
    except:
        continue

print('---------initializing label dictionary--------')
for key in tqdm(img_dict.keys()):
    img_name = img_dict[key]
    lbl_name = img_name[:-4]+'.txt'
    lbl_ann_dict[lbl_name] = []

print('---------getting annotations for each image--------')
for ann in tqdm(anns):
    try:
        img_name = img_dict[ann['image_id']]
        img_path = os.path.join(IMG_DIR, img_name)
        img_file = cv2.imread(img_path, 0)
        img_h, img_w = img_file.shape
        lbl_name = img_name[:-4]+'.txt'
        segmentation = []
        for i, seg in enumerate(ann['segmentation'][0]):
            segmentation.append(seg/ (img_h if i%2==1 else img_w))
        lbl_ann_dict[lbl_name].append([segmentation, ann['category_id']])
    except:
        continue

print('---------writing segmentation in files--------')
for key in tqdm(lbl_ann_dict.keys()):
    lbl_path = os.path.join(LBL_DIR, key)
    lbl_file = open(lbl_path, 'w')
    segs = lbl_ann_dict[key]
    for seg in segs:
        coords, cat = seg[0], seg[1]
        cont = str(int(cat)-1)+' '
        for coord in coords:
            cont+= str(coord)+' '
        lbl_file.write(cont[:-1]+'\n')
    lbl_file.close()
