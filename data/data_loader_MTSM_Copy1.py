# data loader
from __future__ import print_function, division
import glob
import torch
from skimage import io, transform, color
import numpy as np
import random
import math
import matplotlib.pyplot as plt
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
	sum_r = buffer_0[0:56, :, 0].mean()
	sum_g = buffer_0[0:56, :, 1].mean()
	sum_b = buffer_0[0:56, :, 2].mean()
	std_r = buffer_0[0:56, :, 0].std()
	std_g = buffer_0[0:56, :, 1].std()
	std_b = buffer_0[0:56, :, 2].std()
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
		tmpImg[:, :, :, 0] = (buffer[:, :, :, 0] - 0.407) / 0.254
		tmpImg[:, :, :, 1] = (buffer[:, :, :, 1] - 0.413) / 0.260
		tmpImg[:, :, :, 2] = (buffer[:, :, :, 2] - 0.402) / 0.276
	else:
		tmpImg[:, :, :, 0] = (buffer[:, :, :, 0] - 0.169) / 0.140
		tmpImg[:, :, :, 1] = (buffer[:, :, :, 1] - 0.150) / 0.140
		tmpImg[:, :, :, 2] = (buffer[:, :, :, 2] - 0.130) / 0.145

	return tmpImg

class SalObjDataset(Dataset):
	def __init__(self, img_name_list, lbl_name_list):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		# self.transform = transform
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

		buffer = np.empty((16, 256, 256, 3), np.dtype('float'))
		if ((idx % 7500) < 15):        # 0%1875=0,1%1875=1,1876%1875=1
			
			image_0 = cv2.imread(self.image_name_list[idx])
			image_0 = cv2.cvtColor(image_0, cv2.COLOR_BGR2RGB)
			#image_0 = transform.resize(image_0, (self.resize_height, self.resize_width), mode='constant')
			for i in range(16):
				buffer[i, :, :, :] = image_0

			#buffer[6, :, :, :] = image_6            
			#buffer[7, :, :, :] = image_7           
            
		else :

			for i in range(16):
				image = cv2.imread(self.image_name_list[idx-i])
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				buffer[i, :, :, :] = image

		# imname = self.image_name_list[idx]
		imidx = np.array([idx])
		'''
		if ( (idx % 1500) < 5 ):
			if (0 == len(self.label_name_list)):
				label_3 = np.zeros(image.shape)
			else:
				label_3 = io.imread(self.label_name_list[idx])

			label = np.zeros(label_3.shape[0:2])
		else:
			if (0 == len(self.label_name_list)):
				label_3 = np.zeros(image.shape)
			else:
				label_3 = io.imread(self.label_name_list[idx])
		'''
		if (0 == len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])
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

		label = transform.resize(label, (self.resize_height, self.resize_width), mode='constant', order=0,preserve_range=True)

		sample = {'imidx': imidx, 'image': buffer, 'label': label}

		sample = self.Random_Crop(sample)
		sample = self.ToTensorLab(sample)
		return sample

	def Random_Crop(self, sample):
		self.crop_size = (224, 224)
		imidx, buffer, label = sample['imidx'], sample['image'], sample['label']

		if random.random() >= 0.5:
			buffer = buffer[:, :, ::-1, :]  # 纵向镜像，即左右翻转
			label = label[:, ::-1, :]


		h, w = buffer.shape[1:3]
		new_h, new_w = self.crop_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)
		buffer = buffer[:, top: top + new_h, left: left + new_w, :]
		label = label[top: top + new_h, left: left + new_w, :]
		return {'imidx': imidx, 'image': buffer, 'label': label}
	'''
	def transform(snippet):
		snippet = np.concatenate(snippet, axis=-1)
		snippet = torch.from_numpy(snippet).permute(2, 0, 1).contiguous().float()
		snippet = snippet.mul_(2.).sub_(255).div(255)
		snippet = snippet.view(1, -1, 3, snippet.size(1), snippet.size(2)).permute(0, 2, 1, 3, 4)
		return snippet
	
	def ToTensorLab(self, sample):
		imidx, buffer, label = sample['imidx'], sample['image'], sample['label']
		tmpLbl = np.zeros(label.shape)

		if (np.max(label) < 1e-6):
			label = label
		else:
			label = label / np.max(label)
		# with rgb color
		tmpImg = np.zeros((buffer.shape[0], buffer.shape[1], buffer.shape[2], 3))
		# buffer = buffer / np.max(buffer)
		tmpImg = buffer
		tmpImg = tmpImg.transpose((3, 0, 1, 2))
		) / 0.228
		
		tmpImg = torch.from_numpy(tmpImg.copy())
		tmpImg = tmpImg.mul_(2.).sub_(255).div(255)

		tmpLbl[:, :, 0] = label[:, :, 0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		# transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		# tmpImg = tmpImg.transpose((3, 0, 1, 2))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx': torch.from_numpy(imidx), 'image': tmpImg,
				'label': torch.from_numpy(tmpLbl.copy())}
	'''
	def ToTensorLab(self, sample):
		imidx, buffer, label = sample['imidx'], sample['image'], sample['label']
		tmpLbl = np.zeros(label.shape)

		if (np.max(label) < 1e-6):
			label = label
		else:
			label = label / np.max(label)
		# with rgb color
		tmpImg = np.zeros((buffer.shape[0], buffer.shape[1], buffer.shape[2], 3))
		tmpImg = cal_distance(buffer,tmpImg)



		tmpLbl[:, :, 0] = label[:, :, 0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		# transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		tmpImg = tmpImg.transpose((3, 0, 1, 2))
		#buffer = buffer.transpose((3, 0, 1, 2))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx': torch.from_numpy(imidx), 'image': torch.from_numpy(tmpImg.copy()),
				'label': torch.from_numpy(tmpLbl.copy())}

