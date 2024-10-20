import os
from typing import List, Any, Tuple
from KL_loss import KLDLoss1vs1
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.optim as optim
import torchvision.transforms as standard_transforms

import numpy as np
import glob
import os
from visdom import Visdom

from data import SalObjDataset

from model import MTSF

import time
import requests

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ------- 1. define loss function --------

#bce_loss = nn.BCELoss(size_average=True)

dev_name = 'cuda:0'
dev = torch.device(dev_name if torch.cuda.is_available() else "cpu")
print(dev)
kl_loss = KLDLoss1vs1(dev)
bce_loss = nn.BCELoss(size_average=True)

def muti_KL_BCE_loss_fusion(d0, d1, d2, d3, d4,  labels_v):
    #BCE + KL loss

    BCE_loss0 = bce_loss(d0,labels_v)
    KL_loss0 = kl_loss(d0,labels_v)
    loss0 = BCE_loss0 + KL_loss0*0.1

    BCE_loss1 = bce_loss(d1,labels_v)
    KL_loss1 = kl_loss(d1,labels_v)
    loss1 = BCE_loss1 + KL_loss1*0.1
    
    BCE_loss2 = bce_loss(d2,labels_v)
    KL_loss2 = kl_loss(d2,labels_v)
    loss2 = BCE_loss2 + KL_loss2*0.1

    BCE_loss3 = bce_loss(d3,labels_v)
    KL_loss3 = kl_loss(d3,labels_v)
    loss3 = BCE_loss3 + KL_loss3*0.1
    
    BCE_loss4 = bce_loss(d4,labels_v)
    KL_loss4 = kl_loss(d4,labels_v)
    loss4 = BCE_loss4 + KL_loss4*0.1
    '''
   
    # KL loss
    loss0 = kl_loss(d0,labels_v)
    loss1 = kl_loss(d1,labels_v)
    loss2 = kl_loss(d2,labels_v)
    loss3 = kl_loss(d3,labels_v)
    loss4 = kl_loss(d4,labels_v)
    #loss5 = kl_loss(d5,labels_v)
    '''

    loss = loss0 + loss1 + loss2 + loss3 + loss4 
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item()))

    return loss0, loss


# ------- 2. set the directory of training dataset --------

model_name = 'MTSF' #

data_dir = '/media/ailvin/F/DREYE_train'
tra_image_dir = '/media/ailvin/F/DREYE_train/frame_garmin/'
tra_label_dir = '/media/ailvin/F/DREYE_train/frame_saliency/'

image_ext = '.jpg'
label_ext = '.png'


model_dir = os.path.join(os.getcwd(), 'saved_models', model_name + os.sep)

#epoch_num = 100000
epoch_num = 30
#batch_size_train = 12
batch_size_train = 10
batch_size_val = 1
train_num = 0
val_num = 0

#tra_img_name_list = glob.glob(data_dir + tra_image_dir + '*' + image_ext)
tra_img_name_list = glob.glob(tra_image_dir + '*' + image_ext)
tra_img_name_list = sorted(tra_img_name_list)

tra_lbl_name_list = []
'''
for img_path in tra_img_name_list:
    img_name = img_path.split(os.sep)[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    tra_lbl_name_list.append(tra_label_dir + imidx + label_ext)
    #tra_lbl_name_list.append(data_dir + tra_label_dir + imidx + label_ext)
'''
tra_lbl_name_list = glob.glob(tra_label_dir + '*' + label_ext)
tra_lbl_name_list = sorted(tra_lbl_name_list)
print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)
salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    )

salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=60)

# ------- 3. define model --------
# define the net
if(model_name=='MTSF'):
    net = MTSF(pretrained=True)
    print(net)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
ignored_params = list(map(id,net.backbone.parameters()))

base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer = torch.optim.Adam([{'params':base_params},
                              {'params':net.backbone.parameters(),'lr':0.0001}], lr=0.001, weight_decay=2e-7)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma = 0.5)
#net.load_state_dict(torch.load('/home/ailvin/forlunwen/MTSF/saved_models/MTSF2022-06-14_bce_itr_8000_train_0.358060_tar_0.067943.pth')) #用以恢复训练

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 4000 # save the model every 4000 iterations
save_loss = []  
iteration = []  # iterations
save_loss_iterations = [],[]
accumulation_steps = 20

for epoch in range(0, epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):
        #print("inferencing:",tra_img_name_list[i].split(os.sep)[-1])
        ite_num = ite_num + 1  #show batch num
        ite_num4val = ite_num4val + 1

        inputs, labels = data['image'], data['label']
        
        inputs = inputs.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # wrap them in Variable
        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(),
                                                                                        requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        # y zero the parameter gradients
        #optimizer.zero_grad()

        # forward + backward + optimize
        d0, d1, d2, d3, d4 = net(inputs_v)
        #labels_v = float(labels_v.cpu())
        loss2, loss = muti_KL_BCE_loss_fusion(d0, d1, d2, d3, d4,  labels_v)

        loss = loss/accumulation_steps
        loss.backward()
        
        if((i+1)%accumulation_steps)==0:
            optimizer.step()       # update parameters of net
            optimizer.zero_grad()  # reset gradient
        
        # # print statistics
        running_loss += (loss.data.item())*accumulation_steps
        running_tar_loss += loss2.data.item()

        # del temporary outputs and loss
        del d0, d1, d2, d3, d4,  loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
        epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            Todaydate = time.strftime('%Y-%m-%d', time.localtime(time.time()))  #get today date like '2021-11-19'

            torch.save(net.state_dict(), model_dir + model_name+str(Todaydate)+"_bce_itr_%d_train_%3f_tar_%3f.pth" % (ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))
            running_loss = 0.0
            running_tar_loss = 0.0
            #net.train()  # resume train
            ite_num4val = 0
    scheduler.step()  # 更新学习率


