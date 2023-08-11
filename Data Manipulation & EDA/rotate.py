import math
import cv2

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)


def double_flip(file_name):
    file = open(file_name, 'r')
    nf = open(file_name[:-4]+'_180.txt', 'w')
    anns = file.readlines()
    for line in anns:
        ps = line[2:-1].split(' ')
        seg = []
        for i, d in enumerate(ps):
            seg.append(d)
        seg_f = line[:2]
        # print(seg)
        for x, y in pairwise(seg):
            x_, y_ = 1-float(x), 1-float(y)
            seg_f+= str(x_)+' '+str(y_)+' '
        nf.write(seg_f[:-1]+'\n')

def change_label(file_name, angle, w, h):
    if angle==180:
        double_flip(file_name)
        return
    p, q = w/2, h/2
    file = open(file_name, 'r')
    nf = open(file_name[:-4]+'_'+str(angle)+'.txt', 'w')
    anns = file.readlines()
    for line in anns:
        ps = line[2:-1].split(' ')
        seg = []
        for i, d in enumerate(ps):
            seg.append(float(d)*(w if i%2==0 else h))
        seg_f = line[:2]
        # print(seg)
        div_x = w if angle==180 else h
        div_y = h if angle==180 else w
        for x, y in pairwise(seg):
            # print(type(x), type(y))
            cos_r = math.cos(math.radians(angle))
            sin_r = math.sin(math.radians(angle))
            # x_ = x*cos_r - y*sin_r
            # y_ = y*cos_r + x*sin_r
            x_ = q + (x - p) * cos_r - (y - q) * sin_r
            y_ = p + (x - p) * sin_r + (y - q) * cos_r

            seg_f+= str(x_/div_x)+' '+str(y_/div_y)+' '
        nf.write(seg_f[:-1]+'\n')

def rotate_image(file_name, angle):
    rot_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    src = cv2.imread(file_name)
    rot = cv2.rotate(src, rot_map[angle])
    cv2.imwrite(file_name[:-4]+'_'+str(angle)+'.png', rot)



if __name__== '__main__':   
    path = '0a0a4bf1-6345-41ff-82a5-f0d480f55d2f.png'
    
    # Reading an image in default mode
    src = cv2.imread(path)
    
    # Window name in which image is displayed
    # window_name = 'Image'
    
    # Using cv2.rotate() method
    # Using cv2.ROTATE_90_CLOCKWISE rotate
    # by 90 degrees clockwise
    image = cv2.rotate(src, cv2.ROTATE_180)
    
    cv2.imwrite(path[:-4]+'_r.png', image)
    h, w, c = src.shape
    print(image.shape)
    print(src.shape)
    change_label('0a0a4bf1-6345-41ff-82a5-f0d480f55d2f.txt', 180, w, h)