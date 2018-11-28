import Models
import LoadBatches
import glob
import cv2
import numpy as np


def predict_segmentation():
    n_classes = 2
    images_path = 'data/test_2/'
    input_width = 512
    input_height = 512
    epoch_number = 100

    output_path = 'seg_results/'

    m = Models.Unet(n_classes, input_height=input_height, input_width=input_width)

    m.load_weights("results/model_" + str(epoch_number-1) + ".h5")
    m.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

    output_height = 512
    output_width = 512

    images = glob.glob(images_path + "*.jpg") + glob.glob(images_path + "*.png") + glob.glob(images_path + "*.jpeg")
    images.sort()

    colors = [(0, 0, 0), (255, 255, 255)]

    for imgName in images:
        # imgName = imgName.replace('\\', '/')
        outName = imgName.replace(images_path, output_path)
        X = LoadBatches.getImageArr(imgName, input_width, input_height)
        pr = m.predict(np.array([X]))[0]
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((output_height, output_width, 3))
        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        seg_img = cv2.resize(seg_img, (input_width, input_height))
        cv2.imwrite(outName, seg_img)


predict_segmentation()