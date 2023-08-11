import cv2
import numpy as np

# Load the image and annotation file.
image = cv2.imread("images/0abbdca9-eeea-4b04-9ae6-6a1fb8c0fe16.png")
h, w, c = image.shape
# annotation = np.loadtxt("0003bdf1-2486-4408-96dc-140f45010f80_v.txt", dtype=np.int32)
file = open("labels/0abbdca9-eeea-4b04-9ae6-6a1fb8c0fe16.txt", 'r')
anns = file.readlines()
poly = []

def pairwise(iterable):
    "s -> (s0, s1), (s2, s3), (s4, s5), ..."
    a = iter(iterable)
    return zip(a, a)

for line in anns:
    if len(line)<5:
        continue
    ps = line[2:-1].split(' ')
    seg = []
    # print(ps)
    for i, p in enumerate(ps):
        seg.append(int(float(p)*(w if i%2==0 else h)))
    seg_f = []
    for x, y in pairwise(seg):
        seg_f.append([x, y])
    poly.append(np.array(seg_f, dtype=np.int32).reshape((-1, 1, 2)))
    # print(poly[-1])
# Plot the annotation on the image.

color = (255, 0, 0)
thickness = 2
isClosed = True

for points in poly:
    image = cv2.polylines(image, [points], isClosed, color, thickness)

# show image
cv2.imwrite('test_1.png', image)
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Display the image.
# cv2.imshow("Image", image)
# cv2.waitKey(0)
