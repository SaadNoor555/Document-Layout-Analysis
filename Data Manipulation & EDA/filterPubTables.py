# import json
# import os
# from tqdm import tqdm

# def filter_annotations(annotation_file, images_dir, output_file):
#     image_filenames = set(os.listdir(images_dir))

#     with open(annotation_file, 'r') as f:
#         annotations = json.load(f)

#     filtered_images = []
#     filtered_annotations = []

#     for image in tqdm(annotations['images']):
#         if image['file_name'] in image_filenames:
#             filtered_images.append(image)
#             filtered_annotations.extend([ann for ann in annotations['annotations'] if ann['image_id'] == image['id']])

#     annotations['images'] = filtered_images
#     annotations['annotations'] = filtered_annotations

#     with open(output_file, 'w') as f:
#         json.dump(annotations, f)

# annotation_file_path = "/media/abhijit/New Volume/dls2/code/datasets/publy_tables/labels/train/train.json"
# images_dir_path = "/media/abhijit/New Volume/dls2/code/datasets/publy_tables/images/train"
# filter_annotations(annotation_file_path, images_dir_path, "publy_filtered.json")



import json
import os
import cv2
from tqdm import tqdm

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

ANN_DIR = '/media/abhijit/New Volume/dls2/code/datasets/publy_tables/labels/train/train.json'
IMG_DIR = '/media/abhijit/New Volume/dls2/code/datasets/publy_tables/images/train'
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

images_js = []
annoations_js = []
categories_js = cats


print('---------getting image ids--------')
for img in tqdm(imgs):
    img_name = img['file_name']
    img_path = os.path.join(IMG_DIR, img_name)
    if os.path.exists(img_path):
        img_id = img['id']
        img_dict[img_id] = img_name
        images_js.append(img)

print('---------getting annotations for each image--------')
for ann in tqdm(anns):
    try:
        img_name = img_dict[ann['image_id']]
        img_path = os.path.join(IMG_DIR, img_name)
        if os.path.exists(img_path):
            annoations_js.append(ann)
    except:
        print('error')

print('---------writing segmentation in files--------')
annotations = dict()
annotations['categories'] = categories_js
annotations['images'] = images_js
annotations['annotations'] = annoations_js

with open("filteredPublay.json", 'w') as f:
    json.dump(annotations, f)