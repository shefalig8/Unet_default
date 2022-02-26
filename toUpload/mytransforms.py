import numpy as np
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import os
import cv2
from torch.utils.data.sampler import SubsetRandomSampler

# resize
class Resize(object):
    def __init__(self, output_size: tuple) -> None:
        # note output_size needs to be a tuple
        self.output_size = output_size
    
    def __call__(self, img):
        out_img = cv2.resize(img, self.output_size)
        return out_img

    def __repr__(self) -> str:
        return '{},{}'.format(self.output_size[0],self.output_size[1])

#
# Affine
# RandomCrop
# HorizontalFlip
# more rhobust 

# sharpenImag
# 

#  [100, 100]
# resize_fobj = Resize((100,100))


# def resize_img(img, out_size):
#     out_img = cv2.resize(img, output_size)
#     return out_img

# out1= resize_fobj(np.random.rand((20,20))) 
# out = resize_fobj(np.random.rand((300,400))) 

# numpy to tensor
class ToTensor(object):  
    # def __init__(self):
    #     pass  
    def __call__(self, img: np.ndarray)->np.ndarray:
        img = img.transpose((2, 0, 1))
        return img

        
        