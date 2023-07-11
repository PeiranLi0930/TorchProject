import pathlib

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
    Generate image segmentations and mask that to original image
    """

    img_float = skimage.util.img_as_float(img_8bit)
    img_mask = skimage.segmentation.felzenszwalb(img_8bit,
                                                 scale = scale,
                                                 sigma = sigma,
                                                 min_size = min_size)
    img = np.dstack([img_8bit, img_mask]) # the img[:, :, 3] is the segmentation mask

    return(img)

def extract_region(img):
    """
    For each segmented region, extract the smallest rectangle regions covering the smallest region.
    """
    img_segment = img[:, :, 3]
    region = {}

    for y, i in enumerate(img_segment):
        for x, j in enumerate(i):

            if j not in region:
                region[j] = {"up_left_x" : np.Inf,
                             "up_left_y" : np.Inf,
                             "down_right_x" : 0,
                             "down_right_y" : 0,
                             "region" : j}

            if region[j]["up_left_x"] > x:
                region[j]["up_left_x"] = x
            if region[j]["up_left_y"] > y:
                region[j]["up_left_y"] = y
            if region[j]["down_right_x"] < x:
                region[j]["down_right_x"] = x
            if region[j]["down_right_y"] < y:
                region[j]["down_right_y"] = y

    copied_region_dict = copy(region)

    for key in region.keys():
        if (region[key]["down_right_x"] == region[key]["up_left_x"] or
        region[key]["down_right_y"] == region[key]["up_left_y"]):
            del copied_region_dict[key]

    return copied_region_dict

def plt_rectangle(plt, label, x1, y1, x2, y2, color = "yellow", alpha = 0.5):
    linewidth = 3
    if type(label) == list:
        linewidth = len(label) * 3 + 2
        label = ""

    plt.text(x1, y1, label, fontsize = 20, backgroundcolor = color, alpha = alpha)
    plt.plot([x1, x1], [y1, y2], linewidth = linewidth, color = color, alpha = alpha)
    plt.plot([x2, x2], [y1, y2], linewidth = linewidth, color = color, alpha = alpha)
    plt.plot([x1, x2], [y1, y1], linewidth = linewidth, color = color, alpha = alpha)
    plt.plot([x1, x2], [y2, y2], linewidth = linewidth, color = color, alpha = alpha)

def calc_texture_gradient(img):
    """
    Calculate texture gradient.
    The original Selective Search algo used Gaussian Derivative for 8 orientation.
    Here, we use LBP.
    """
    ret = np.zeros(img.shape[:3])
    for c in (0, 1, 2):
        ret[:, :, c] = skimage.feature.local_binary_pattern(img[:, :, c], 8, 1.0)

    return ret

def to_hsv(img):
    """
    IMG from RGB to HSV. (Hue, Saturation, Value)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv

def generate_hist(img, minhist = 0, maxhist = 1):
    """
        calculate colour histogram for each region

        the size of output histogram will be BINS * COLOUR_CHANNELS(3)

        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf]

        extract HSV

        len(hist) = BINS * 3
        hist[:BINS] = [0, 10, 20, 0,...,0] meaning that
           there are 10 pixels that have values between (maxhist - minhist)/BINS*1 and (maxhist - minhist)/BINS*2
           there are 20 pixels that have values between (maxhist - minhist)/BINS*2 and (maxhist - minhist)/BINS*3

    """

    BINS = 25 # length of unit hist
    hist = np.array([])
    for channel in range(3):
        c = img[:, :, channel]
        # hist will return back a tuple (histogram, bin_edges)
        # e.g. (array([2037, 2002, 2016, 2041, 1972, 2023, 2022, 1994, 2010, 2047, 2073,
        #         1933, 2088, 1976, 2059, 1986, 1980, 2021, 2028, 1990, 1984, 1914,
        #         1936, 2077, 1967]),
        #  array([0.  , 0.04, 0.08, 0.12, 0.16, 0.2 , 0.24, 0.28, 0.32, 0.36, 0.4 ,
        #         0.44, 0.48, 0.52, 0.56, 0.6 , 0.64, 0.68, 0.72, 0.76, 0.8 , 0.84,
        #         0.88, 0.92, 0.96, 1.  ], dtype=float32))
        hist = np.concatenate([hist] + [np.histogram(c, BINS, (minhist, maxhist))[0]])

    hist = hist / len(img) # normalize
    return hist

def augmented_regions_with_histogram_info(texture_grad, img, region_dict: dict, hsv, tex_trad):
    for k, v in list(region_dict.items()):
        masked_pixels = hsv[img[:, :, 3] == k]
        region_dict[k]["size"] = len(masked_pixels / 4) # 4 channels
        region_dict[k]["hist_channel"] = generate_hist(masked_pixels, minhist = 0, maxhist = 1)
        region_dict[k]["hist_texture"] = generate_hist(texture_grad[img[:, :, 3] == k], minhist =
        0, maxhist = 2 ** 8 - 1)

    return region_dict

def extract_neighbours(regions):
    def intersect(a, b) -> bool:
        """
        Determine whether there are intersection between two windows
        """
        if (a["up_left_x"] < b["up_left_x"] < a["down_right_x"] and a["up_left_y"] < b[
            "up_left_y"] < a["down_right_y"]) or \
                (a["up_left_x"] < b["down_right_x"] < a["down_right_x"] and a["up_left_y"] < b[
                    "up_left_y"] < a["down_right_y"]) or \
                (a["up_left_x"] < b["down_right_x"] < a["down_right_x"] and a["up_left_y"] < b[
                    "down_right_y"] < a["down_right_y"]) or \
                (a["up_left_x"] < b["up_left_x"] < a["down_right_x"] and a["up_left_y"] < b[
                    "down_right_y"] < a["down_right_y"]):
            return True
        return False

    region_dict_list = list(regions.items()) # [("": ), ("": ), ("": ), ...]
    neighbors = []

    for current, a in enumerate(region_dict_list[:-1]):
        for b in region_dict_list[current + 1:]:
            if intersect(a[1], b[1]):
                neighbors.append((a, b))

    return neighbors


if __name__ == '__main__':
    img = cv2.imread(os.path.join("image", "example_id1.JPG"), cv2.IMREAD_COLOR)

    segmented_img = image_seg(img)
    R = extract_region(segmented_img)

    figsize = (20, 20)
    plt.figure(figsize = figsize)
    plt.imshow(img[:, :, :3])
    for item, color in zip(R.values(), sns.xkcd_rgb.values()):
        x1 = item["up_left_x"]
        y1 = item["up_left_y"]
        x2 = item["down_right_x"]
        y2 = item["down_right_y"]
        label = item["region"]
        plt_rectangle(plt, label, x1, y1, x2, y2, color = color)
    plt.show()

    # plt.figure(figsize = figsize)
    # plt.imshow(img[:, :, 3])
    # for item, color in zip(R.values(), sns.xkcd_rgb.values()):
    #     x1 = item["min_x"]
    #     y1 = item["min_y"]
    #     x2 = item["max_x"]
    #     y2 = item["max_y"]
    #     label = item["labels"][0]
    #     plt_rectangle(plt, label, x1, y1, x2, y2, color = color)
    # plt.show()


    # np.random.seed(4)
    # list_path = os.listdir("./image")
    # total_num = len(list_path)
    # rand_img_path = np.random.choice(list_path, 1)
    # img_8bit = cv2.imread(os.path.join("image", rand_img_path[0]), cv2.IMREAD_COLOR)
    #
    # plt.imshow(to_hsv(img_8bit))
    # plt.show()

    # img = img_8bit[:, :, :3]
    # img = calc_texture_gradient(img)
    # plt.imshow(img)
    # plt.show()

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







