import imp


import numpy as np
# import matplotlib.pyplot as plt
import cv2


def show_img(img, win_name, key=1):    
    cv2.namedWindow(win_name,0)
    cv2.imshow(win_name,img)
    cv2.waitKey(key)

def read_image(impath):
    img = cv2.imread(impath)
    # img = cv2.cvtColor(img_in, )
    
    # reads image [H, W, C] opencv to numpy happens default
    # numpy image: H x W x C
    # torch image: C x H x W, (D,H,W)
    # 
    if img is not None:
        return img
    else:
        print('Imread error in path', impath)


# img  = read_image(train_imgs_list[0])
# print(img.shape)
# show_img(img, 'input')

# # resize_image        
# resize_fptr = Resize((224,224)) #  Assume as callable pointer
# rimg = resize_fptr(img)
# print(resize_fptr)
# print(rimg.shape)
# show_img(rimg, 'resized_input')

# tensor_image  
# tensor_img = ToTensor()
# timg = tensor_img(img)
# print(timg.shape)
# show_img(timg, 'to_tensor')

# quit(0)
# # train, val  -> one loader
# # test -> labels

# datasset * trans 
# 1000 * 5
# 5000