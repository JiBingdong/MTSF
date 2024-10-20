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
import cv2
import numpy as np
from PIL import Image
import glob
from KL_loss import KLDLoss1vs1, cc_numeric, SIM_numeric, NSS, AUC_Judd,InfoGain,AUC_Borji

from data import SalObjDataset_test_DReveVE

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
    AUC_Judd_loss = AUC_Judd(d0, fixlabels_v)
    IG_loss = InfoGain(d0, fixlabels_v, baselabels)
    AUC_Borji_loss = AUC_Borji(d0, fixlabels_v)
    #IG_loss = 0
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
        image = io.imread(image_name_list[batchsize*i_test+k])

        imo = im.resize((1920,1080),resample=Image.BILINEAR)

        pb_np = np.array(imo)
        image = Image.fromarray(image)
        image = image.resize((1920,1080),resample=Image.BILINEAR)
        aaa = img_name.split(".")
        bbb = aaa[0:-1]
        imidx = bbb[0]
        for i in range(1,len(bbb)):
            imidx = imidx + "." + bbb[i]
        #image.save(d_dir + imidx + '.jpg')
        imo.save(d_dir+imidx+'.png')

def main():

    # --------- 1. get image path and name ---------
    model_name='MTSF'

    image_dir = '/media/ailvin/F/DREYE_test/frame_garmin'

    prediction_dir = '/home/ailvin/桌面/实验记录/prediction/Methods-DREYE/'
    tra_label_dir = '/media/ailvin/F/DREYE_test/frame_saliency/'
    
    model_dir = '/home/ailvin/forlunwen/MTSF/saved_models/'
    img_name_list = glob.glob(image_dir + os.sep + '*'+'.jpg')
    img_name_list = sorted(img_name_list)
    #img_name_list = img_name_list[7500:]

    tra_lbl_name_list = glob.glob(tra_label_dir + '*' +'.png')
    tra_lbl_name_list = sorted(tra_lbl_name_list)
    #print(img_name_list)
    img_name_list = img_name_list[112200:]
    tra_lbl_name_list = tra_lbl_name_list[112200:]
    a = len(img_name_list)
    print(a)
    # --------- 2. dataloader ---------
    #1. dataloader

    test_salobj_dataset = SalObjDataset_test_DReveVE(img_name_list=img_name_list,lbl_name_list=tra_lbl_name_list)
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=40)

    # --------- 3. model define ---------
    if(model_name=='MTSF'):
        print("...load MTSF---189.1 MB")
        net = MTSF(pretrained=False)

    if torch.cuda.is_available():
        #model_dir = model_dir + '/u2net.pth'
        model_dir = model_dir + 'MTSF2022-06-23_bce_itr_4000_train_1.064740_tar_0.205953.pth'
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
        print('cuda:0')
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval() #可以防止由于测试和训练的batchsize不同而导致的错误
    KL_running_loss = 0.0
    CC_running_loss = 0.0
    SIM_running_loss = 0.0
    NSS_running_loss = 0.0
    AUC_Judd_running_loss = 0.0
    IG_running_loss = 0.0
    AUC_Borji_running_loss = 0.0
    test_num = 0

    # 创建对象，用于视频的写出.每次都要改名
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWrite_frame = cv2.VideoWriter(r'/home/ailvin/USB_Image/videos/qinliangh_test_DREYEVE_8.mp4', fourcc, 25,
                                       (1344, 768))
    videoWrite_SOE = cv2.VideoWriter(r'/home/ailvin/USB_Image/videos/qinliangh_test_DREYEVE_MTSF_8.mp4', fourcc, 25,
                                     (1344, 768))
    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        test_num = i_test +1

        img_name = img_name_list[i_test * 1]
        vid_index = int(img_name[40:42]) #/media/ailvin/F/DREYE_test/frame_garmin/43frame_004213.jpg
        #print(vid_index)
        #print("inferencing:",img_name)
        print("inferencing:%3i/%3i" % (int(i_test * 1), int(a)))
        img = cv2.imread(img_name)
        origin_img = cv2.resize(img, (256, 256))

        inputs_test,labels, fixlabels = data_test['image'], data_test['label'], data_test['fixlabel']
        inputs_test = inputs_test.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)
        fixlabels = fixlabels.type(torch.FloatTensor)
        #baselabels = baselabels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test,labels_v,fixlabels_v = Variable(inputs_test.cuda()),Variable(labels.cuda()),Variable(fixlabels.cuda())
        else:
            inputs_test,labels_v,fixlabels_v = Variable(inputs_test),Variable(labels),Variable(fixlabels)

        d1,d2,d3,d4,d5 = net(inputs_test)
        #d1 = net(inputs_test)

        baseline_path = '/media/ailvin/F/mean_saliency/' + str(vid_index) +'.png'
        baselabels_3 = cv2.imread(baseline_path)
        baselabels_3 = cv2.cvtColor(baselabels_3, cv2.COLOR_BGR2RGB)
        baselabels_3 = cv2.resize(baselabels_3, (256, 256))
        #baselabels = np.zeros((256,256,1))
        #print(baselabels.shape)
        baselabels = baselabels_3[:,:,0]
        baselabels = baselabels/255

        baselabels = baselabels[:, :, np.newaxis]
        #print(baselabels.shape)
        baselabels = baselabels.transpose((2, 0, 1))


        KL_loss, CC_loss, SIM_loss, NSS_loss, AUC_Judd_loss,IG_loss,AUC_Borji_loss = muti_bce_loss_fusion(d1, labels_v, fixlabels_v, baselabels)
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

        print(" KL_loss: %3f ,CC_loss: %3f, SIM_loss: %3f, NSS_loss: %3f, AUC_Judd_loss: %3f, IG_loss: %3f,AUC_Borji_loss: %3f"% (
        KL_running_loss / test_num, CC_running_loss / test_num, SIM_running_loss / test_num,
        NSS_running_loss / test_num, AUC_Judd_running_loss / test_num, IG_running_loss / test_num, AUC_Borji_running_loss / test_num))

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

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

        # save results to test_results folder
        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        #batch_size = pred.size(0)

        #if 0.4<KL_loss < 1.0:
           # save_output(img_name_list, pred, prediction_dir, i_test)

        #save_output(img_name_list, pred, prediction_dir, i_test)
        del d1, d2, d3, d4, d5

        #del d1


if __name__ == "__main__":
    main()
