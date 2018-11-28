import cv2
import numpy as np
import glob
import os

DIR = "/home/buiduchanh/WorkSpace/Unet/Unet-for-Person-Segmentation/data/real_data/valid/process_1"
DES = "/home/buiduchanh/WorkSpace/Unet/Unet-for-Person-Segmentation/data/real_data/valid/process_2"
imglist = sorted(glob.glob('{}/*'.format(DIR)))
for img in imglist:
    basename = os.path.splitext(os.path.basename(img))[0]
    print(basename)
    # exit()
    im = cv2.imread(img)
    label = np.zeros_like(im, dtype='uint8')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)

    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j] != 0:
                label[i, j] = 1
    cv2.imwrite('{}/{}.png'.format(DES, basename), label)
