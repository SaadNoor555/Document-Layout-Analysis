from shapely import geometry
import numpy as np
import cv2
import os
from tqdm import tqdm

def shrink_poly(coords, factor=0.02):
    # code from nathan
    xs = [i[0] for i in coords]
    ys = [i[1] for i in coords]
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)

    min_corner = geometry.Point(min(xs), min(ys))
    max_corner = geometry.Point(max(xs), max(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner)*factor

    # assert abs(shrink_distance - center.distance(max_corner)) < 0.0001

    my_polygon = geometry.Polygon(coords)
    my_polygon_shrunken = my_polygon.buffer(-shrink_distance)
    xs, ys = my_polygon_shrunken.exterior.xy
    shrunken_coords = []
    for x, y in zip(xs, ys):
        shrunken_coords.extend([x, y])
    return shrunken_coords

def get_pts(pts, h, w):
    points = []
    for i in range(0, len(pts), 2):
        x, y = int(float(pts[i])*w), int(float(pts[i+1])*h)
        points.append((x, y))
    return points

def get_coords(h, w, line):
    pts = line[2:-1].split(' ')
    points = np.array(get_pts(pts, h, w))
    return points

def change_ann(img, lbl, factor):
    h, w, c = img.shape
    f = open(lbl, 'r')
    lines = f.readlines()
    f.close()
    fw = open(lbl, 'w')
    for line in lines:
        if line[0]!='1':
            fw.write(line)
        else:
            try:
                coords = get_coords(h, w, line)
                shrunken = shrink_poly(coords, factor)
                for i in range(0, len(shrunken), 2):
                    shrunken[i] = shrunken[i]/w
                    shrunken[i+1] = shrunken[i+1]/h
                shrunken_str = [str(pt) for pt in shrunken]
                newline = '1 '+' '.join(shrunken_str)+'\n'
                fw.write(newline)
            except:
                fw.write(line)
            

IMG_DIR = 'datasets/badlad/badlad/images/train'
LBL_DIR = 'datasets/badlad/badlad/labels/train'
imglist = os.listdir(IMG_DIR)
for imgname in tqdm(imglist):
    img_path = os.path.join(IMG_DIR, imgname)
    lbl = imgname[:-4]+'.txt'
    lbl_path = os.path.join(LBL_DIR, lbl)
    img = cv2.imread(img_path)
    change_ann(img, lbl_path, 0.02)
