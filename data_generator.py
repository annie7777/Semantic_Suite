import cv2
import glob
import os
import numpy as np

from skimage.measure import label, regionprops

groundtruth_folder = 'gt'

if not os.path.exists(groundtruth_folder):
    os.mkdir(groundtruth_folder)

images = glob.glob('dataset157/images/*.jpg')
images.extend(glob.glob('dataset157/images/*.bmp'))

for index, file in enumerate(images):
    basename = os.path.basename(file)
    print(basename)
    img = cv2.imread(file)
    mask = cv2.imread(os.path.join('dataset157/masks', basename), 0)
    ret, thresh = cv2.threshold(mask,10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3), np.uint(8))
    dilation = cv2.dilate(thresh, kernel, iterations = 1)


    label_image = label(dilation)
    # print(np.unique(label_image))
    regions = regionprops(label_image)
    print(len(regions))

    for i, region in enumerate(regionprops(label_image)):

        if region.area < 50:
            continue
        else:
            firstpoint = region.coords[0]
            label_number = label_image[firstpoint[0], firstpoint[1]]
            minr,minc,maxr,maxc = region.bbox

            h = maxr-minr
            w = maxc-minc

            cc_img = np.zeros((img.shape[0], img.shape[1]))
            cc_img[label_image==label_number] = 255

            cc_img = img[minr:maxr, minc:maxc]
            cv2.imwrite(os.path.join(groundtruth_folder,'{}_00{}.jpg'.format(basename,i)), cc_img)

            # cv2.imshow('c', cc_img)
            # cv2.waitKey(1000)

            if i == 100:
                break


    if index == 2:
        break