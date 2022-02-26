from typing import DefaultDict
import numpy as np
from sklearn import metrics
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import os
import cv2
import csv
from torch.utils.data.sampler import SubsetRandomSampler
from mytransforms import Resize,ToTensor
from imageUtils import read_image,show_img
from utils import read_csv_from_camvid,color_to_mask,show_mask_imag
import torch as th


# TODO
# move gloabal to CamVID
# move the transforms to
class CamVid(Dataset):
    def __init__(self, impaths, labels, target_transform, label_file=None):
        self.impaths = impaths
        self.labels = labels
        self.label_map = label_file 
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.impaths)

# CamVid[i]
# 
    def __getitem__(self, idx):
        image = read_image(self.impaths[idx])
        label = read_image(self.labels[idx])
        # image = read_image(r'E:\CodeSpace\python\data_sets\CamVid\train\0001TP_009210.png')
        # label_in = read_image(r'E:\CodeSpace\python\data_sets\CamVid\train_labels\0001TP_009210_L.png')

        label_in = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
        mask = color_to_mask( label_in, self.label_map) #  h,w,nclasses

        # color_map = {
        # 'Void':(0,0,0),
        # 'Car': (64, 0, 128), # 0
        # 'Pedestrian': (64, 64, 0), # 1
        # 'Sky':	(128,128,128), # 2
        # 'Archway':(192,0,128),
        # 'SUVPickupTruck':(64,128,192)
        # }

        # show_mask_imag(mask, color_map)

        # print('mask shape ',mask.shape) 
        # print('impaht  ',self.impaths[idx])
        # print('mask shape ',mask.shape) 
        # mask = np.argmax(mask, axis=2 )  # H,W, N


        # print(' after mask shape ',mask.shape) 

        # t = [ np.any(mask == x) for x in range(6)]
        # print(' Label Maps ', t)

        # print(' AFTER image shape ',image.shape)
        # print('AFTER label shape ',mask.shape)


        if self.target_transform:
            # Resize torch.Size([1, 4, 724, 964]),ToTensor
            image = self.target_transform(image)
            # to Tensor
            mask = transforms.ToTensor()(mask)
            # B,W,H

        # print('AFTER transform ',mask.shape)
        return image, mask


def tensor_to_numpy(timg):
    return timg.transpose((1, 2, 0))
    

if __name__ == "__main__":

    camvid_root = r'E:\CodeSpace\python\data_sets\CamVid'
    train_img_root = camvid_root+'\\train'
    train_lbl_root = camvid_root+'\\train_labels'
    val_imgs_list = camvid_root+'\\val'
    val_lbl_root = camvid_root+'\\val_labels'
    color_code_map_file = camvid_root + '\\class_dict.csv'

    color_map = {
    'Void':(0,0,0),
    'Car': (64, 0, 128), # 0
    'Pedestrian': (64, 64, 0), # 1
    'Sky':	(128,128,128), # 2
    'Archway':(192,0,128),
    'SUVPickupTruck':(64,128,192)
    }

    def list_files_in_dir(root_path):
        return [os.path.join(root_path,impaths) for impaths in os.listdir(root_path)]  

    train_imgs_list   = list_files_in_dir(train_img_root)
    train_labels_list = list_files_in_dir(train_lbl_root)

    val_imgs_list =  list_files_in_dir(val_imgs_list)
    val_labels_list = list_files_in_dir(val_lbl_root)

    # label_img = read_image(train_labels_list[0])
    # color_code_map_file = 
    # color_map = read_csv_from_camvid(color_code_map_file)

    camvid_transforms = transforms.Compose([
                                #transforms.RandomRotation(30),  # data augmentations are great
                                #transforms.RandomResizedCrop(224),  # but not in this case of map tiles
                                # transforms.Resize( (720//2 ,960//2)),
                                transforms.ToPILImage(),
                                transforms.Resize([920,1160]),
                                transforms.ToTensor(),
                                #transforms.Normalize([0.485, 0.456, 0.406], # PyTorch recommends these but in this
                                #                     [0.229, 0.224, 0.225]) # case I didn't get good results
                                ])

    train_dataset = CamVid(train_imgs_list, train_labels_list, camvid_transforms,color_map )

    trainloader = th.utils.data.DataLoader(train_dataset, sampler=SubsetRandomSampler([x for x in range(len(train_imgs_list))]), batch_size=1)
    

    for img, label in trainloader:
        print(img.size())
        print(label.size())

        # print(label*255)
        batch_size = label.shape[0]
        for b in range(batch_size):
            print('befor .numpy',label[b].shape)
            label_np = label[b].numpy()
            print('after  .numpy ',label_np.shape)
            label_np = np.transpose(label_np, (1, 2, 0))  # convert from Tensor image
            print('befor argmax',label_np.shape)
            label_np = np.argmax(label_np, axis=2 )
            print('after argmax',label_np.shape)

            label_cv = np.zeros_like(label_np,np.uint8)

            t = [ np.any(label_np == x) for x in range(6)]

            print(' All class res ', t)
            label_cv[:,:,0] = 42*label_np[:,:,0]
            
            print(label_cv.shape)
            show_img(label_cv, "Label_View{}".format(b),0)
        quit(0)