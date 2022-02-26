import csv
from collections import defaultdict
from typing import DefaultDict
import numpy as np
from imageUtils import show_img


def read_csv_from_camvid(filename):
    map_dict = DefaultDict(str)
    # map_dict = read_csv_from_camvid(color_code_map_file)
    with open(filename, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for line in csv_reader:
            map_dict[line[0]] = tuple([int(line[1]), int(line[2]), int(line[3]) ])
        return map_dict


def color_to_mask(rgb_image, colormap ):
    num_classes = len(colormap)
    # print(num_classes)
    # h,w,3 
    shape = rgb_image.shape[:2]+(num_classes,)
    mask = np.zeros( shape, dtype=np.uint8 )
    #  h,w,nclasses
    for i, pair in enumerate(colormap.items()):
        mask[:,:,i] = np.all( rgb_image == pair[1], axis=2)
    return mask


def show_mask_imag(mask, color_map):
    for i, color in enumerate(color_map.items()):
        timg = 255*mask[:,:,i]
        show_img(timg,color[0])
        # print('loop : ',color[0], color[1])
        # cv2.imwrite('{}.jpg'.format(color[0]),timg)
