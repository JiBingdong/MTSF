import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# import torch.optim as optim
import json
import numpy as np
from PIL import Image
import glob
import cv2
from KL_loss import KLDLoss1vs1, cc_numeric, SIM_numeric,NSS,AUC_Judd_batch,AUC_Borji_batch,AUC_shuffled

from data import SalObjDataset_test_DADA
#from data_loader_test_DADA_Norm import SalObjDataset
from model import MTSF # full size version 173.6 MB

# normalize the predicted SOD probability map

dev_name = 'cuda:0'
dev = torch.device(dev_name if torch.cuda.is_available() else "cpu")
print(dev)
kl_loss = KLDLoss1vs1(dev)

def muti_bce_loss_fusion(d0, labels_v,fixlabels_v):
    KL_loss0 = kl_loss(d0,labels_v)
    CC_loss0 = cc_numeric(labels_v,d0)
    SIM_loss = SIM_numeric(d0, labels_v)
    NSS_loss = NSS(d0, fixlabels_v)
    AUC_Judd_loss = AUC_Judd_batch(d0, fixlabels_v)
    #AUC_Shuffeld_loss = AUC_shuffled(d0, fixlabels_v, baselabels)
    AUC_Borji_loss = AUC_Borji_batch(d0, fixlabels_v)

    #print("KL_l0: %3f ,CC_l0: %3f, SIM_l0:%3f , NSS_l0:%3f, AUC_Judd_l0:%3f ,AUC_Borji_l0:%3f"%(KL_loss0.data.item(),CC_loss0,SIM_loss,NSS_loss,AUC_Judd_loss,AUC_Borji_loss))

    return KL_loss0,CC_loss0,SIM_loss,NSS_loss,AUC_Judd_loss,AUC_Borji_loss


def normPRED(d):
    batchsize = d.size(0)
    for k in range(batchsize):
        ma = torch.max(d[k])
        mi = torch.min(d[k])
        d[k] = (d[k]-mi)/(ma-mi)

    return d

def save_output(image_name_list,pred,d_dir,i_test):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    batchsize = predict.size(0)
    h = predict.size(1)
    w = predict.size(2)
    for k in range(batchsize):
        im_k = np.zeros((h, w))
        im_k = predict_np[k, :, :]
        im = Image.fromarray(im_k*255).convert('RGB')
        img_name = image_name_list[batchsize*i_test+k].split(os.sep)[-1]
        image = io.imread(image_name_list[batchsize*i_test+k])

        imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
        image = Image.fromarray(image)
        pb_np = np.array(imo)
        image_name = image_name_list[batchsize*i_test+k]
        if image_name[37] == '/':
            vid_index_1 = image_name[36]     # 1
            vid_index_2 = image_name[38:41]  # 002
        else:
            vid_index_1 = image_name[36:38]  # 14
            vid_index_2 = image_name[39:42]  #002
        # /home/ailvin/forlunwen/DADA_dataset/1/001/images/0002.jpg
        #vid_index = image_name[36:42]  # 1/001/ or 14/002


        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        #image.save(d_dir+vid_index_1+'_'+vid_index_2+'_'+imidx+'.jpg')
        imo.save(d_dir+vid_index_1+'_'+vid_index_2+'_'+ imidx + '.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='MTSF'
    prediction_dir = '/home/ailvin/桌面/实验记录/prediction/DADA-2000/'

    root = '/home/ailvin/forlunwen/DADA_dataset/'
    #valid_imgs = [json.loads(line) for line in open(root + 'val_file.json')]
    test_imgs = [json.loads(line) for line in open(root + 'test_file.json')]

    test_img_name_list = []
    test_lbl_name_list = []
    test_fix_name_list = []
    #video_list = valid_imgs[0]
    video_list = test_imgs[0]
    DADA_path = '/home/ailvin/forlunwen/DADA_dataset/'
    # /media/ailvin/F/DADA-2000/DADA-2000/DADA_dataset/1/001/images/0002.jpg
    for i in range(len(video_list)):
        video_name1 = video_list[i][0][0]  # 1
        video_name2 = video_list[i][0][1]  # 001
        image_name_list = glob.glob(DADA_path + str(video_name1) + '/' + str(video_name2) + '/' + 'images/' + '*' + '.jpg')
        label_name_list = glob.glob(DADA_path + str(video_name1) + '/' + str(video_name2) + '/' + 'maps/' + '*' + '.jpg')
        fixation_name_list = glob.glob(DADA_path + str(video_name1) + '/' + str(video_name2) + '/' + 'fixation/' + '*' + '.png')
        image_name_list = sorted(image_name_list)
        label_name_list = sorted(label_name_list)
        fixation_name_list = sorted(fixation_name_list)
        test_img_name_list += image_name_list
        test_lbl_name_list += label_name_list
        test_fix_name_list += fixation_name_list

    #test_img_name_list = test_img_name_list[57500:]
    #test_lbl_name_list = test_lbl_name_list[57500:]
    #test_fix_name_list = test_fix_name_list[57500:]
    print("---")
    a = len(test_img_name_list)
    print("test images: ", a)
    print("train labels: ", len(test_lbl_name_list))
    print("train fixation labels: ", len(test_fix_name_list))
    print("---")

    salobj_dataset = SalObjDataset(
        img_name_list=test_img_name_list,
        lbl_name_list=test_lbl_name_list,
        fix_name_list=test_fix_name_list
    )
    test_salobj_dataloader = DataLoader(salobj_dataset, batch_size=3, shuffle=False, num_workers=40)

    # --------- 3. model define ---------
    if(model_name=='MTSF'):
        print("...load MTSF---189.1 MB")
        net = MTSF(pretrained=False)

    if torch.cuda.is_available():
        #model_dir = model_dir + '/u2net.pth'
        model_dir = '/home/ailvin/forlunwen/MTSF/saved_models/' + 'MTSF2022-07-07_bce_itr_28000_train_7.271334_tar_1.248853.pth'
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
        print('cuda:0')
    net.eval() #可以防止由于测试和训练的batchsize不同而导致的错误
    KL_running_loss = 0.0
    CC_running_loss = 0.0
    SIM_running_loss = 0.0
    NSS_running_loss = 0.0
    #IG_running_loss = 0.0
    AUC_Judd_running_loss = 0.0
    AUC_Borji_running_loss = 0.0

    KL_running_loss_d1 = 0.0
    KL_running_loss_d2 = 0.0
    KL_running_loss_d3 = 0.0
    KL_running_loss_d4 = 0.0
    CC_running_loss_d1 = 0.0
    CC_running_loss_d2 = 0.0
    CC_running_loss_d3 = 0.0
    CC_running_loss_d4 = 0.0
    SIM_running_loss_d1 = 0.0
    SIM_running_loss_d2 = 0.0
    SIM_running_loss_d3 = 0.0
    SIM_running_loss_d4 = 0.0
    NSS_running_loss_d1 = 0.0
    NSS_running_loss_d2 = 0.0
    NSS_running_loss_d3 = 0.0
    NSS_running_loss_d4 = 0.0
    AUC_Judd_running_loss_d1 = 0.0
    AUC_Judd_running_loss_d2 = 0.0
    AUC_Judd_running_loss_d3 = 0.0
    AUC_Judd_running_loss_d4 = 0.0
    AUC_Borji_running_loss_d1 = 0.0
    AUC_Borji_running_loss_d2 = 0.0
    AUC_Borji_running_loss_d3 = 0.0
    AUC_Borji_running_loss_d4 = 0.0
    test_num = 0
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        test_num = i_test +1
        img_name = test_img_name_list[i_test]
        #vid_index = int(img_name[0:2])
        # print(vid_index)
        image_name = test_img_name_list[3*i_test]
        print(image_name)
        print("inferencing:%3i/%3i"%( int(i_test*3),int(a)))

        inputs_test,labels, fixlabels = data_test['image'], data_test['label'], data_test['fixlabel']
        inputs_test = inputs_test.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        fixlabels = fixlabels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test,labels_v,fixlabels_v = Variable(inputs_test.cuda()),Variable(labels.cuda()),Variable(fixlabels.cuda())
        else:
            inputs_test,labels_v,fixlabels_v = Variable(inputs_test),Variable(labels),Variable(fixlabels)

        d0,d1,d2,d3,d4 = net(inputs_test)
        #d1 = net(inputs_test)
        '''
        baseline_path = '/home/ailvin/forlunwen/CDNN_code_data/traffic_dataset/saliencydata/mean_saliency/' + str(vid_index) + '.png'
        baselabels_3 = cv2.imread(baseline_path)
        baselabels_3 = cv2.cvtColor(baselabels_3, cv2.COLOR_BGR2RGB)
        baselabels_3 = cv2.resize(baselabels_3, (256, 256))
        # baselabels = np.zeros((256,256,1))
        # print(baselabels.shape)
        baselabels = baselabels_3[:, :, 0]
        baselabels = baselabels / 255

        baselabels = baselabels[:, :, np.newaxis]
        # print(baselabels.shape)
        baselabels = baselabels.transpose((2, 0, 1))
        '''

        KL_loss, CC_loss, SIM_loss, NSS_loss, AUC_Judd_loss, AUC_Borji_loss = muti_bce_loss_fusion(d0,labels_v,fixlabels_v)
        KL_loss_1, CC_loss_1, SIM_loss_1, NSS_loss_1, AUC_Judd_loss_1, AUC_Borji_loss_1 = muti_bce_loss_fusion(d1, labels_v,
                                                                                                   fixlabels_v)
        KL_loss_2, CC_loss_2, SIM_loss_2, NSS_loss_2, AUC_Judd_loss_2, AUC_Borji_loss_2 = muti_bce_loss_fusion(d2, labels_v,
                                                                                                   fixlabels_v)
        KL_loss_3, CC_loss_3, SIM_loss_3, NSS_loss_3, AUC_Judd_loss_3, AUC_Borji_loss_3 = muti_bce_loss_fusion(d3, labels_v,
                                                                                                   fixlabels_v)
        KL_loss_4, CC_loss_4, SIM_loss_4, NSS_loss_4, AUC_Judd_loss_4, AUC_Borji_loss_4 = muti_bce_loss_fusion(d4,
                                                                                                               labels_v,
                                                                                                               fixlabels_v)
        if str(CC_loss) == 'nan':
            CC_loss = 1.0
            KL_running_loss += KL_loss.data.item()
            CC_running_loss += CC_loss
            SIM_running_loss += SIM_loss
            NSS_running_loss += NSS_loss
            AUC_Judd_running_loss += AUC_Judd_loss
            AUC_Borji_running_loss += AUC_Borji_loss
        else:
            KL_running_loss += KL_loss.data.item()
            CC_running_loss += CC_loss
            SIM_running_loss += SIM_loss
            NSS_running_loss += NSS_loss
            AUC_Judd_running_loss += AUC_Judd_loss
            #IG_running_loss += IG_loss
            AUC_Borji_running_loss += AUC_Borji_loss
        print(
            " KL_loss_d0: %3f ,CC_loss_d0: %3f, SIM_loss_d0: %3f, NSS_loss_d0: %3f, AUC_Judd_loss_d0: %3f, AUC_Borji_loss_d0: %3f" % (
                KL_running_loss / test_num, CC_running_loss / test_num, SIM_running_loss / test_num,
                NSS_running_loss / test_num, AUC_Judd_running_loss / test_num,
                AUC_Borji_running_loss / test_num))

        KL_running_loss_d1 += KL_loss_1.data.item()
        CC_running_loss_d1 += CC_loss_1
        SIM_running_loss_d1 += SIM_loss_1
        NSS_running_loss_d1 += NSS_loss_1
        AUC_Judd_running_loss_d1 += AUC_Judd_loss_1
        AUC_Borji_running_loss_d1 += AUC_Borji_loss_1
        print(
            " KL_loss_d1: %3f ,CC_loss_d1: %3f, SIM_loss_d1: %3f, NSS_loss_d1: %3f, AUC_Judd_loss_d1: %3f, AUC_Borji_loss_d1: %3f" % (
                KL_running_loss_d1 / test_num, CC_running_loss_d1 / test_num, SIM_running_loss_d1 / test_num,
                NSS_running_loss_d1 / test_num, AUC_Judd_running_loss_d1 / test_num,
                AUC_Borji_running_loss_d1 / test_num))

        KL_running_loss_d2 += KL_loss_2.data.item()
        CC_running_loss_d2 += CC_loss_2
        SIM_running_loss_d2 += SIM_loss_2
        NSS_running_loss_d2 += NSS_loss_2
        AUC_Judd_running_loss_d2 += AUC_Judd_loss_2
        AUC_Borji_running_loss_d2 += AUC_Borji_loss_2
        print(
            " KL_loss_d2: %3f ,CC_loss_d2: %3f, SIM_loss_d2: %3f, NSS_loss_d2: %3f, AUC_Judd_loss_d2: %3f, AUC_Borji_loss_d2: %3f" % (
                KL_running_loss_d2 / test_num, CC_running_loss_d2 / test_num, SIM_running_loss_d2 / test_num,
                NSS_running_loss_d2 / test_num, AUC_Judd_running_loss_d2 / test_num,
                AUC_Borji_running_loss_d2 / test_num))

        KL_running_loss_d3 += KL_loss_3.data.item()
        CC_running_loss_d3 += CC_loss_3
        SIM_running_loss_d3 += SIM_loss_3
        NSS_running_loss_d3 += NSS_loss_3
        AUC_Judd_running_loss_d3 += AUC_Judd_loss_3
        AUC_Borji_running_loss_d3 += AUC_Borji_loss_3
        print(
            " KL_loss_d3: %3f ,CC_loss_d3: %3f, SIM_loss_d3: %3f, NSS_loss_d3: %3f, AUC_Judd_loss_d3: %3f, AUC_Borji_loss_d3: %3f" % (
                KL_running_loss_d3 / test_num, CC_running_loss_d3 / test_num, SIM_running_loss_d3 / test_num,
                NSS_running_loss_d3 / test_num, AUC_Judd_running_loss_d3 / test_num,
                AUC_Borji_running_loss_d3 / test_num))

        KL_running_loss_d4 += KL_loss_4.data.item()
        CC_running_loss_d4 += CC_loss_4
        SIM_running_loss_d4 += SIM_loss_4
        NSS_running_loss_d4 += NSS_loss_4
        AUC_Judd_running_loss_d4 += AUC_Judd_loss_4
        AUC_Borji_running_loss_d4 += AUC_Borji_loss_4
        print(
            " KL_loss_d4: %3f ,CC_loss_d4: %3f, SIM_loss_d4: %3f, NSS_loss_d4: %3f, AUC_Judd_loss_d4: %3f, AUC_Borji_loss_d4: %3f" % (
                KL_running_loss_d4 / test_num, CC_running_loss_d4 / test_num, SIM_running_loss_d4 / test_num,
                NSS_running_loss_d4 / test_num, AUC_Judd_running_loss_d4 / test_num,
                AUC_Borji_running_loss_d4 / test_num))

        # normalization
        #pred = d0[:,0,:,:]
        #pred = normPRED(pred)

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        #batch_size = pred.size(0)
        '''
        if 0.01< KL_loss <0.5:
            save_output(test_img_name_list ,pred,prediction_dir,i_test)
        '''
        #vid_index = image_name[39:42]
        #if vid_index == '013':
        #save_output(test_img_name_list, pred, prediction_dir, i_test)
        del d0,d1,d2,d3,d4
        #del d1

if __name__ == "__main__":
    main()
