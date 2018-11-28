import cv2
import numpy as np
import glob
import os

DIR = '/home/buiduchanh/WorkSpace/Unet/Unet-for-Person-Segmentation/data/real_data/valid/val_anno'
DES = '/home/buiduchanh/WorkSpace/Unet/Unet-for-Person-Segmentation/data/real_data/valid/process_1'
imglist = sorted(glob.glob('{}/*'.format(DIR)))
for img in imglist:
    basename = os.path.splitext(os.path.basename(img))[0]
    basename = basename[:-5]
    img = cv2.imread(img)
    print(basename)
    for i in range (len(img)):
        for j in range (len(img[i])):
            # print(img[i][j])
            B = img[i][j][0]
            G = img[i][j][1]
            R = img[i][j][2]
            if R > 200:
                img[i][j] = np.array([255, 255, 255])
            else :
                img[i][j] = np.array([0, 0, 0])

    cv2.imwrite('{}/{}.png'.format(DES,basename),img)







# img = '/home/buiduchanh/WorkSpace/Unet/image-segmentation-keras/test.png'
# img = cv2.imread(img)
# w, h ,c = img.shape
# ann_img = np.zeros((w,h,3)).astype('uint8')
# ann_img = img[:,:,0]
# print(ann_img.shape)
# cv2.imwrite('anno_test.png',ann_img)

