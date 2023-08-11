import os 
import cv2
from tqdm import tqdm
import random

NUM_CLASSES = 4

def check_labels(file):
    f = open(file, 'r')
    lines = f.readlines()
    flg_dict = {}
    for i in range(NUM_CLASSES):
        flg_dict[i] = False
    for l in lines:
        if len(l)>0:
            flg_dict[int(l[0])] = True
    return flg_dict

def count_freq(dir):
    cnt_dict = {}
    for i in range(NUM_CLASSES):
        cnt_dict[i] = 0
    for file in tqdm(os.listdir(dir)):
        file_path = os.path.join(dir, file)
        flg_dict = check_labels(file_path)
        
        for key in flg_dict.keys():
            if flg_dict[key]:
                cnt_dict[key]+=1
    
    return cnt_dict
    
def sep_val(dir, val_dict):
    files = os.listdir(dir)
    random.shuffle(files)
    valfile = []
    td = {}
    for i in range(NUM_CLASSES):
        td[i] = 0
    for file in tqdm(files):
        file_path = os.path.join(dir, file)
        flg_dict = check_labels(file_path)
        taken = False
        for i in range(NUM_CLASSES):
            if flg_dict[i] and val_dict[i]>0:
                taken = True
        
        if taken:
            for i in range(NUM_CLASSES):
                if flg_dict[i]:
                    val_dict[i] -= 1
                    td[i] += 1
            valfile.append(file)
    f = open('val_badlad.txt', 'w')
    for line in valfile:
        f.write(line+'\n')
    print('*****class wise distribution of val files*****')
    print(td)
    print('number of val files =', len(valfile))
    return valfile

def move_val(lbl_src, lbl_des, img_src, img_des, files):
    for file in tqdm(files):
        img_name = file[:-4]+'.png'
        img_path_src = os.path.join(img_src, img_name)
        img_path_des = os.path.join(img_des, img_name)
        lbl_path_src = os.path.join(lbl_src, file)
        lbl_path_des = os.path.join(lbl_des, file)
        os.rename(img_path_src, img_path_des)
        os.rename(lbl_path_src, lbl_path_des)


LBL_DIR = 'datasets/badlad/badlad/labels/train'
IMG_DIR = 'datasets/badlad/badlad/images/train'
LBL_VAL = 'datasets/badlad/badlad/labels/val'
IMG_VAL = 'datasets/badlad/badlad/images/val'
print('*****counting frequency of each class*****')
cnt_dict = count_freq(LBL_DIR)
print(cnt_dict)
val_dict = {}
for key in cnt_dict.keys():
    val_dict[key] = cnt_dict[key]//5
print('*****choosing val files*****')
valfiles = sep_val(LBL_DIR, val_dict)
print('*****moving chosen val files*****')
move_val(LBL_DIR, LBL_VAL, IMG_DIR, IMG_VAL, valfiles)
