import torch as th
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from torchvision import datasets, transforms, models

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import datetime, os
import torch.utils.tensorboard as tb
from torch.utils.tensorboard import SummaryWriter

# custom
import torch.optim as optim
from model_unet import UNet
from torch.utils.data.sampler import SubsetRandomSampler
from mytransforms import Resize,ToTensor
from imageUtils import read_image,show_img
from utils import read_csv_from_camvid,color_to_mask
from dataset import CamVid


camvid_root = r'E:\CodeSpace\python\data_sets\CamVid'

train_img_root = camvid_root+'\\train'
train_lbl_root = camvid_root+'\\train_labels'
val_imgs_list = camvid_root+'\\val'
val_lbl_root = camvid_root+'\\val_labels'
color_code_map_file = camvid_root + '\\class_dict.csv'

# check if CUDA is available
train_on_gpu = th.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')



def train_model(model, trainloader, valloader, logger, epochs = 100):
    model.train() 
    steps = 0
    with open('Train Log.txt', 'w, ')
    #add training accuracy
    for epoch in range(epochs):
        train_loss = 0
        valid_loss = 0
        accuracy_train = 0
        # 10000 
        # 1 epcoh 
        for inputs, labels in trainloader:
            steps += 1
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            logps = model.forward(inputs) #y_predicted
            # 
            # 
            logps = F.softmax(logps, dim=1)
            # logps = th.argmax(logps, axis=1)

            labels = th.argmax(labels, axis=1)

            # print('logps',logps.shape)
            # print('labels',labels.shape)

            loss = criterion(logps, labels) #y_predicted, y---->loss function
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() #train loss
            print("Epoch {} training loss {}, total loss {}".format(epoch,loss.item(),train_loss))

        # evaluate 
        train_loss = train_loss/len(trainloader.sampler)
        # keep it,
        # 
        logger.add_scalar('Avg loss/train', train_loss, global_step=epoch)
        # IOU, loss 10 , 20  ===> [x] ==> 
    ###########Validation######## 
        accuracy_val = 0

    # mock test
        for inputs, labels in valloader:
            if train_on_gpu:
                inputs, labels = inputs.cuda(), labels.cuda()
            logps = model.forward(inputs) #op=y_pred
            logps = F.softmax(logps,dim=1)
	    ##
	    labels = th.argmax(labels, axis=1)
            # NCWH
            # print('logps',logps.shape)
            # print('labels',labels.shape)
            batch_loss = criterion(logps, labels)  #loss fuction
            valid_loss += batch_loss.item()*inputs.size(0)
	    print("Epoch {} validation loss {}, total loss {}".format(epoch,batch_loss.item(),valid_loss))
            #  IOU
            # rest 
        valid_loss = valid_loss/len(valloader.sampler)
        logger.add_scalar('Avg loss/val', valid_loss, global_step=epoch)
        

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f},\tTrain meanIOU:{:.6f},  \tValidation Loss: {:.6f},\tValidation meanIOU:{:.6f}'.format(
            epoch, train_loss,accuracy_train, valid_loss,accuracy_val))
   
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            #torch.save(model.state_dict(), 'model_cifar.pt')
            checkpoint = {
                'epochs': epochs,
                'model': model,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict()
                }
            th.save(checkpoint, 'checkpoint.pth')
            valid_loss_min = valid_loss   
    
    writer.close()



if __name__ == "__main__":

    def list_files_in_dir(root_path):
        return [os.path.join(root_path,impaths) for impaths in os.listdir(root_path)]  

    # 
    model = UNet(3,6)

    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        model.cuda()

    # specify loss function (categorical cross-entropy)
    criterion = nn.CrossEntropyLoss()

    # specify optimizer, 10^-4
    optimizer = optim.SGD(model.parameters(), lr=0.0001)

    writer = SummaryWriter()
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    logger = tb.SummaryWriter(logdir + '/test',flush_secs=1)  #1

    print_every = 10
    valid_loss_min = np.Inf
    #train_losses, test_losses = [], []

    train_imgs_list   = list_files_in_dir(train_img_root)
    train_labels_list = list_files_in_dir(train_lbl_root)

    val_imgs_list =  list_files_in_dir(val_imgs_list)
    val_labels_list = list_files_in_dir(val_lbl_root)

    label_img = read_image(train_labels_list[0])
    # color_code_map_file = 
    color_map = {
    'Void':(0,0,0),
    'Car': (64, 0, 128), # 0
    'Pedestrian': (64, 64, 0), # 1
    'Sky':	(128,128,128), # 2
    'Archway':(192,0,128),
    'SUVPickupTruck':(64,128,192)
    }
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
    val_dataset = CamVid(val_imgs_list, val_labels_list, camvid_transforms, color_map)

    trainloader = th.utils.data.DataLoader(train_dataset, sampler=SubsetRandomSampler([x for x in range(len(train_imgs_list))]), batch_size=32)
    valloader = th.utils.data.DataLoader(val_dataset, sampler=SubsetRandomSampler([x for x in range(len(val_imgs_list))]), batch_size=32)
    train_model(model, trainloader, valloader, logger, epochs = 500)