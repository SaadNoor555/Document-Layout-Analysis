import os
from tqdm import tqdm

LBL_DIR = 'datasets/badlad/tables/labels/train'
LBL_VAL = 'datasets/publynet/labels/val'

def change_cls(dir):
    for file in tqdm(os.listdir(dir)):
        file_path = os.path.join(dir, file)
        f = open(file_path, 'r')
        file_cont = f.readlines()
        f.close()
        f = open(file_path, 'w')
        for line in file_cont:
            if len(line)<1:
                continue
            new_line = str(int(line[0])-1)+line[1:]+'\n'
            f.write(new_line)

def remove_newline(dir):
    for file in tqdm(os.listdir(dir)):
        file_path = os.path.join(dir, file)
        f = open(file_path, 'r')
        file_cont = f.readlines()
        f.close()
        f = open(file_path, 'w')
        for line in file_cont:
            if line=='\n':
                continue
            new_line = line
            f.write(new_line)
remove_newline(LBL_DIR)
# remove_newline(LBL_VAL)
