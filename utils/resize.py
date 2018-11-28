import os
import glob
import cv2 as cv 
Dir = '/home/buiduchanh/WorkSpace/Unet/Unet-for-Person-Segmentation/data/real_data/valid/rust_anno_val'
Des = '/home/buiduchanh/WorkSpace/Unet/Unet-for-Person-Segmentation/data/real_data/valid/val_anno'

def resize_with_pad(image, height, width):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv.copyMakeBorder(image, top , bottom, left, right, cv.BORDER_CONSTANT, value=BLACK)

    resized_image = cv.resize(constant, (height, width))
    
    return resized_image 


imglist = sorted(glob.glob('{}/*'.format(Dir)))
for img in imglist:
    print(img)
    basename = os.path.splitext(os.path.basename(img))[0]
    image  = cv.imread(img)
    imgpad = resize_with_pad(image, 512 , 512)
    cv.imwrite('{}/{}.jpg'.format(Des,basename), imgpad)