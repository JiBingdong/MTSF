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
from KL_loss import KLDLoss1vs1, cc_numeric, SIM_numeric,NSS,AUC_Judd_batch,AUC_Borji_batch,InfoGain

from data import SalObjDataset_test_CDNN

from model import MTSF # full size version 173.6 MB

# normalize the predicted SOD probability map

dev_name = 'cuda:0'
dev = torch.device(dev_name if torch.cuda.is_available() else "cpu")
print(dev)
kl_loss = KLDLoss1vs1(dev)

def muti_bce_loss_fusion(d0, labels_v,fixlabels_v,baselabels):
    KL_loss0 = kl_loss(d0,labels_v)
    CC_loss0 = cc_numeric(labels_v,d0)
    SIM_loss = SIM_numeric(d0, labels_v)
    NSS_loss = NSS(d0, fixlabels_v)
    AUC_Judd_loss = AUC_Judd_batch(d0, fixlabels_v)
    #IG_loss = InfoGain(d0, fixlabels_v, baselabels)
    AUC_Borji_loss = AUC_Borji_batch(d0, fixlabels_v)
    IG_loss = 0
    print("KL_l0: %3f ,CC_l0: %3f, SIM_l0:%3f , NSS_l0:%3f, AUC_Judd_l0:%3f ,IG_l0:%3f,AUC_Borji_l0:%3f"%(KL_loss0.data.item(),CC_loss0,SIM_loss,NSS_loss,AUC_Judd_loss,IG_loss,AUC_Borji_loss))

    return KL_loss0,CC_loss0,SIM_loss,NSS_loss,AUC_Judd_loss,IG_loss,AUC_Borji_loss

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
        image = io.imread('/home/ailvin/forlunwen/CDNN_code_data/traffic_dataset/traffic_videos/'+image_name_list[batchsize*i_test+k])

        imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
        image = Image.fromarray(image)
        pb_np = np.array(imo)
        video_name = image_name_list[batchsize*i_test+k][0:2]
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]

        print(imidx)
        image.save(d_dir + video_name + '_' + imidx + '.jpg')
        imo.save(d_dir+video_name+'_'+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='MTSF'
    prediction_dir = '/home/ailvin/桌面/实验记录/prediction/TDV/'

    root = '/home/ailvin/forlunwen/CDNN_code_data/traffic_dataset/traffic_frames/'
    #train_imgs = [json.loads(line) for line in open(root + 'train.json')]

    #valid_imgs = [json.loads(line) for line in open(root + 'valid.json')]
    test_imgs = [json.loads(line) for line in open(root + 'test.json')]
    test_img_name_list = test_imgs
    test_img_name_list = sorted(test_img_name_list)
    test_img_name_list = test_img_name_list[7000:]

    print("---")
    a = len(test_img_name_list)
    print("test images: ", a)

    print("---")

    salobj_dataset = SalObjDataset_test_CDNN(
        img_name_list=test_img_name_list
    )
    test_salobj_dataloader = DataLoader(salobj_dataset, batch_size=1, shuffle=False, num_workers=40)

    # --------- 3. model define ---------
    if(model_name=='MTSF'):
        print("...load MTSF---189.1 MB")
        net = MTSF(pretrained=False)

    if torch.cuda.is_available():
        #model_dir = model_dir + '/u2net.pth'
        model_dir = '/home/ailvin/forlunwen/MTSF/saved_models/' + 'MTSF2022-06-14_bce_itr_8000_train_0.358060_tar_0.067943.pth'
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
        print('cuda:0')
    net.eval() #可以防止由于测试和训练的batchsize不同而导致的错误
    KL_running_loss = 0.0
    CC_running_loss = 0.0
    SIM_running_loss = 0.0
    NSS_running_loss = 0.0
    IG_running_loss = 0.0
    AUC_Judd_running_loss = 0.0
    AUC_Borji_running_loss = 0.0
    test_num = 0
    # 创建对象，用于视频的写出.每次都要改名
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWrite_frame = cv2.VideoWriter(r'/media/ailvin/TOSHIBA EXT/data/demo/New_RGB_Channel/test_CDNN_2.mp4', fourcc, 25, (1344, 768))
    videoWrite_SOE = cv2.VideoWriter(r'/media/ailvin/TOSHIBA EXT/data/demo/New_RGB_Channel/MTSF_test_CDNN_2.mp4', fourcc, 25,(1344, 768))

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):
        test_num = i_test +1
        img_name = test_img_name_list[i_test]
        #print(img_name)
        vid_index = int(img_name[0:2])
        # print(vid_index)
        print("inferencing:%3i/%3i"%( int(i_test*1),int(a)))
        img_path = test_img_name_list[i_test]

        data_path = '/home/ailvin/forlunwen/CDNN_code_data/traffic_dataset/traffic_videos/'
        img = cv2.imread(data_path+img_path)
        origin_img = cv2.resize(img, (224, 224))
        #origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)

        inputs_test,labels, fixlabels = data_test['image'], data_test['label'], data_test['fixlabel']
        inputs_test = inputs_test.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        fixlabels = fixlabels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test,labels_v,fixlabels_v = Variable(inputs_test.cuda()),Variable(labels.cuda()),Variable(fixlabels.cuda())
        else:
            inputs_test,labels_v,fixlabels_v = Variable(inputs_test),Variable(labels),Variable(fixlabels)

        d1,d2,d3,d4,d5 = net(inputs_test)
        #d1 = net(inputs_test)
        baseline_path = '/home/ailvin/forlunwen/CDNN_code_data/traffic_dataset/saliencydata/mean_saliency/' + str(vid_index) + '.png'
        baselabels_3 = cv2.imread(baseline_path)
        baselabels_3 = cv2.cvtColor(baselabels_3, cv2.COLOR_BGR2RGB)
        baselabels_3 = cv2.resize(baselabels_3, (224, 224))
        # baselabels = np.zeros((256,256,1))
        # print(baselabels.shape)
        baselabels = baselabels_3[:, :, 0]
        baselabels = baselabels / 255

        baselabels = baselabels[:, :, np.newaxis]
        # print(baselabels.shape)
        baselabels = baselabels.transpose((2, 0, 1))

        KL_loss, CC_loss, SIM_loss, NSS_loss, AUC_Judd_loss, IG_loss, AUC_Borji_loss = muti_bce_loss_fusion(d1,
                                                                                                            labels_v,
                                                                                                            fixlabels_v,
                                                                                                            baselabels)
        if str(CC_loss) == 'nan':
            CC_loss = 0.0
            KL_running_loss += KL_loss.data.item()
            CC_running_loss += CC_loss
            SIM_running_loss += SIM_loss
            NSS_running_loss += NSS_loss
            AUC_Judd_running_loss += AUC_Judd_loss
            IG_running_loss += IG_loss
            AUC_Borji_running_loss += AUC_Borji_loss
        else:
            KL_running_loss += KL_loss.data.item()
            CC_running_loss += CC_loss
            SIM_running_loss += SIM_loss
            NSS_running_loss += NSS_loss
            AUC_Judd_running_loss += AUC_Judd_loss
            IG_running_loss += IG_loss
            AUC_Borji_running_loss += AUC_Borji_loss

        print(
            " KL_loss: %3f ,CC_loss: %3f, SIM_loss: %3f, NSS_loss: %3f, AUC_Judd_loss: %3f, IG_loss: %3f,AUC_Borji_loss: %3f" % (
                KL_running_loss / test_num, CC_running_loss / test_num, SIM_running_loss / test_num,
                NSS_running_loss / test_num, AUC_Judd_running_loss / test_num, IG_running_loss / test_num,
                AUC_Borji_running_loss / test_num))

        
        sal_map = d1.cpu().detach().numpy()[0]
        sal_map = sal_map[0, :, :]
        sal_map = np.clip(sal_map, 0, 1) * 255  # normalized
        sal_map = sal_map.astype(np.uint8)  # type
        sal_heatmap = cv2.applyColorMap(sal_map, cv2.COLORMAP_JET)  # 此处的三通道热力图是cv2使用GBR排列
        origin_img = np.asarray(origin_img)
        origin_img_1 = cv2.addWeighted(origin_img, 1.0, sal_heatmap, 0.6, 0)  # img + sal
        origin_img_1 = cv2.resize(origin_img_1, (1344, 768))
        origin_img = cv2.resize(origin_img, (1344, 768))
        cv2.imshow("origin_img", origin_img)
        cv2.imshow("origin_img_1", origin_img_1)


        # 保存视频
        videoWrite_frame.write(origin_img)
        videoWrite_SOE.write(origin_img_1)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):  # 按下q(quit)键，程序退出
            break
            cv2.destroyAllWindows()

        del d1,d2,d3,d4,d5
        #del d1

if __name__ == "__main__":
    main()
