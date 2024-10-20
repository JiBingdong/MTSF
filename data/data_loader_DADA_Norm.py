# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import cv2
#==========================dataset load==========================
def cal_distance(buffer, tmpImg):
		# 预定义用以储存均值和标准差
	'''
	sum_r = 0
	sum_g = 0
	sum_b = 0
	count = 0
	std_r = 0
	std_g = 0
	std_b = 0
	'''
	tmpImg = tmpImg
	buffer = buffer / np.max(buffer)
	buffer_0 = buffer[0, :, :, :]
	#print(buffer_0.shape)
	sum_r = buffer_0[0:64, :, 0].mean()
	sum_g = buffer_0[0:64, :, 1].mean()
	sum_b = buffer_0[0:64, :, 2].mean()
	std_r = buffer_0[0:64, :, 0].std()
	std_g = buffer_0[0:64, :, 1].std()
	std_b = buffer_0[0:64, :, 2].std()
	node = np.array([sum_r,sum_g,sum_b,std_r,std_g,std_b])
	"""
    计算两个向量之间的欧式距离
    :param node1:
    :param node2:
    :return:
    """
	node1 = np.array([0.63891935,0.66413138,0.67330827,0.22834228,0.22917138,0.2449475]) # 簇1，白天，前1/4的簇中心
	node2 = np.array([0.14315021,0.12962922,0.13871429,0.10003495,0.09572037,0.0875226])  # 簇2，夜晚，前1/4的簇中心
	dis_cluster1 = np.sqrt(np.sum(np.square(node - node1)))
	dis_cluster2 = np.sqrt(np.sum(np.square(node - node2)))
	if dis_cluster1 <= dis_cluster2:
		tmpImg[:, :, :,0] = (buffer[:, :, :,0] - 0.407) / 0.254
		tmpImg[:, :, :,0] = (buffer[ :, :, :,1] - 0.413) / 0.260
		tmpImg[:, :, :,1] = (buffer[:, :, :, 2] - 0.402) / 0.276
	else:
		tmpImg[:, :, :,0] = (buffer[ :, :, :,0] - 0.169) / 0.140
		tmpImg[:, :, :,1] = (buffer[ :, :, :,1] - 0.150) / 0.140
		tmpImg[ :, :, :,2] = (buffer[ :, :, :,2] - 0.130) / 0.145

	return tmpImg

class SalObjDataset(Dataset):
	def __init__(self, img_name_list,lbl_name_list):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.resize_height = 256  #320
		self.resize_width = 256  #320
		#self.crop_size = 224  #288

        #self.Random_Crop=Random_Crop
        #self.ToTensorLab=ToTensorLab

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])
		image_name = self.image_name_list[idx]
		# /media/ailvin/F/DADA-2000/DADA-2000/DADA_dataset/1/001/images/0002.jpg
		# or /media/ailvin/F/DADA-2000/DADA-2000/DADA_dataset/14/002/images/0349.jpg
		#/home/ailvin/forlunwen/DADA_dataset/1/001/images/0002.jpg
		vid_index = image_name[36:42]  #1/001/ or 14/002
		frame_index = image_name[49:53]  #0002

		buffer = np.empty((16, 256, 256, 3), np.dtype('float32'))

		for i in range(16):
			image_i_name = self.image_name_list[idx - i]
			vid_index_i = image_i_name[36:42]
			if vid_index == vid_index_i:
				image_0 = cv2.imread(image_i_name)
				image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)
				image_0 = transform.resize(image_0, (self.resize_height, self.resize_width), mode='constant')
				buffer[i, :, :, :] = image_0
				num = i
			else:
				image_num_name = self.image_name_list[idx - num]
				#image_path = os.path.join(self.data_path, image_num_name)
				image_0 = cv2.imread(image_num_name)
				image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)
				image_0 = transform.resize(image_0, (self.resize_height, self.resize_width), mode='constant')
				buffer[i, :, :, :] = image_0


		# imname = self.image_name_list[idx]
		imidx = np.array([idx])
		'''
		if image_name[54] =='/':
			lable_name = image_name[0:55] +'maps/' +image_name[62:]
		else:
			lable_name = image_name[0:56] + 'maps/' + image_name[63:]
		'''
		# label_3 = io.imread(self.label_name_list[idx])
		label_3 = cv2.imread(self.label_name_list[idx])
		label_3 = cv2.cvtColor(label_3, cv2.COLOR_BGR2RGB)

		label_3 = transform.resize(label_3, (self.resize_height, self.resize_width), mode='constant', order=0,
								 preserve_range=True)
		label = np.zeros(label_3.shape[0:2])
		if (3 == len(label_3.shape)):
			label = label_3[:, :, 0]
		elif (2 == len(label_3.shape)):
			label = label_3

		image = buffer[0,:,:,:]
		#image = image.squeeze(0)
		if (3 == len(image.shape) and 2 == len(label.shape)):
			label = label[:, :, np.newaxis]
		elif (2 == len(image.shape) and 2 == len(label.shape)):
			image = image[:, :, np.newaxis]
			label = label[:, :, np.newaxis]

		#label = transform.resize(label, (self.resize_height, self.resize_width), mode='constant', order=0,preserve_range=True)

		sample = {'imidx': imidx, 'image': buffer, 'label': label}

		   
		sample = self.ToTensorLab(sample)
		sample = self.Random_Crop(sample)
		return sample

	def Random_Crop(self, sample):
		self.crop_size = (224, 224)
		imidx, buffer, label = sample['imidx'], sample['image'], sample['label']

		if random.random() >= 0.5:
			buffer = buffer[:, :, :, ::-1]  # 纵向镜像，即左右翻转
			label = label[:, :, ::-1]


		h, w = buffer.shape[2:4]
		new_h, new_w = self.crop_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)
		buffer = buffer[:, :, top: top + new_h, left: left + new_w]
		label = label[:, top: top + new_h, left: left + new_w]
        
		return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(buffer.copy()),
				'label': torch.from_numpy(label.copy())}
		#return {'imidx': imidx, 'image': buffer, 'label': label}

	def ToTensorLab(self, sample):
		imidx, buffer, label = sample['imidx'], sample['image'], sample['label']
		tmpLbl = np.zeros(label.shape)

		if (np.max(label) < 1e-6):
			label = label
		else:
			label = label / np.max(label)
		# with rgb color
		tmpImg = np.zeros((buffer.shape[0], buffer.shape[1], buffer.shape[2], 3))

		tmpImg[:, :, :, 0] = (buffer[:, :, :, 0] - 0.319) / 0.212
		tmpImg[:, :, :, 1] = (buffer[:, :, :, 1] - 0.316) / 0.216
		tmpImg[:, :, :, 2] = (buffer[:, :, :, 2] - 0.302) / 0.228
		#tmpImg = cal_distance(buffer,tmpImg)   #需要将它移到随即裁减前



		tmpLbl[:, :, 0] = label[:, :, 0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		# transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((3, 0, 1, 2))
		#buffer = buffer.transpose((3, 0, 1, 2))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx': imidx, 'image': tmpImg, 'label': tmpLbl}
        #return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg.copy()),'label': torch.from_numpy(tmpLbl.copy())}

