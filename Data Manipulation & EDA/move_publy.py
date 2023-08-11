import os
from tqdm import tqdm

DATA_DIR = 'datasets/tables/images/train'
PUBLY_DIR = 'datasets/publy_tables/images/train'
# BADLAD_DIR = 'datasets/badlad/badlad/images/train'

def move_files(files, ext, src_fold, dest_fold):
    for file in tqdm(files):
        if file[-len(ext):]==ext:
            src = os.path.join(src_fold, file)
            dest = os.path.join(dest_fold, file)
            os.rename(src, dest)
        
# publy_files = os.listdir(PUBLY_DIR)
data_files = os.listdir(DATA_DIR)

# print('Moving badlad images from publy....')
# move_files(publy_files, '.png', PUBLY_DIR, BADLAD_DIR)

print('moving publy images from dataset...')
move_files(data_files, '.jpg', DATA_DIR, PUBLY_DIR)