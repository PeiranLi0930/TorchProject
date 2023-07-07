import matplotlib.pyplot as plt
import skimage
import os
import numpy as np
import seaborn as sns
import pandas as pd
import scipy.misc
import skimage.segmentation
import skimage.feature
from copy import copy
import cv2

def image_seg(img_8bit, scale = 1.0, sigma = 0.8, min_size = 50):
    """

    Args:
        img_8bit:
        scale:
        sigma:
        min_size:

    Returns: The masked original image (0-255)

    """

    img_float = skimage.util.img_as_float(img_8bit)
    img_mask = skimage.segmentation.felzenszwalb(img_float,
                                                 scale = scale,
                                                 sigma = sigma,
                                                 min_size = min_size)
    img = np.dstack([img_8bit, img_mask]) # the img[:, :, 3] is the segmentation mask

    return(img)


def extract_img(img):
    """
    For each segmented region, extract smallest rectangle regions covering the smallest region.

    Args:
        img: (H, W, C)
        N channel = [R, G, B, L]

    Returns:

    """
    img_segment = img[:, :, 3]
    region = {}

    for y, i in enumerate(img_segment):
        for x, j in enumerate(i):

            if j not in region:
                region[j] = {"down_right_x" : np.Inf,
                             "down_right_y" : np.Inf,
                             "up_left_x" : 0,
                             "up_left_y" : 0,
                             "region" : j}

            if region[j]["down_right_x"] > x:
                region[j]["down_right_x"] = x
            if region[j]["down_right_y"] > y:
                region[j]["down_right_y"] = y
            if region[j]["up_left_x"] < x:
                region[j]["up_left_x"] = x
            if region[j]["up_left_y"] < y:
                region[j]["up_left_y"] = y

            copied_region_dict = copy(region)

    for key in region.keys():
        if (region[key]["down_right_x"] == region[key]["up_left_x"] or
        region[key]["down_right_y"] == region[key]["up_left_y"]):
            del copied_region_dict[key]

    return copied_region_dict

if __name__ == '__main__':
    Config = [1.0, 0.8, 500]

    np.random.seed(4)
    list_path = os.listdir("./image")
    total_num = len(list_path)
    rand_img_path = np.random.choice(list_path, 1)

    img_8bit = cv2.imread(os.path.join("image", rand_img_path[0]), cv2.IMREAD_COLOR)
    print(img_8bit.shape)
    seged_img = image_seg(img_8bit, *Config)
    print(seged_img[:, :, 3])




    # for img in rand_img_path:
    #     img_8bit = cv2.imread(os.path.join("image", img), cv2.IMREAD_COLOR)
    #     seged_img = image_seg(img_8bit, *Config)
    #
    #     fig = plt.figure(figsize = (15, 30))
    #
    #     ax = fig.add_subplot(1, 2, 1)
    #     ax.imshow(img_8bit)
    #     ax.set_title("original image")
    #     ax = fig.add_subplot(1, 2, 2)
    #     ax.imshow(seged_img[:, :, 3]) # the segmentation output
    #     ax.set_title("skimage.segmentation.felzenszwalb, N unique region = {}".format(
    #         len(np.unique(seged_img[:, :, 3]))))
    #
    # plt.show()

    # Region Extraction







