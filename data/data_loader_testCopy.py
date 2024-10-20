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

#==========================dataset load==========================

class SalObjDataset(Dataset):
	def __init__(self, img_name_list, lbl_name_list):
		# self.root_dir = root_dir
		# self.image_name_list = glob.glob(image_dir+'*.png')
		# self.label_name_list = glob.glob(label_dir+'*.png')
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		# self.transform = transform
		self.resize_height = 160  #320
		self.resize_width = 160  #320
		self.crop_size = 144  #288

        #self.Random_Crop=Random_Crop
        #self.ToTensorLab=ToTensorLab

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self, idx):

		# image = Image.open(self.image_name_list[idx])#io.imread(self.image_name_list[idx])
		# label = Image.open(self.label_name_list[idx])#io.imread(self.label_name_list[idx])
		image = io.imread(self.image_name_list[idx])


		buffer = np.empty((6, 160, 160, 3), np.dtype('float'))
		if ((idx % 1500) < 5):        # 0%1875=0,1%1875=1,1876%1875=1
			'''
			a = idx + 7
			image_0 = io.imread(self.image_name_list[a])
			image_0 = transform.resize(image_0, (self.resize_height, self.resize_width), mode='constant')

			image_1 = io.imread(self.image_name_list[a - 1])
			image_1 = transform.resize(image_1, (self.resize_height, self.resize_width), mode='constant')

			image_2 = io.imread(self.image_name_list[a - 2])
			image_2 = transform.resize(image_2, (self.resize_height, self.resize_width), mode='constant')

			image_3 = io.imread(self.image_name_list[a - 3])
			image_3 = transform.resize(image_3, (self.resize_height, self.resize_width), mode='constant')
            
			image_4 = io.imread(self.image_name_list[a - 4])
			image_4 = transform.resize(image_4, (self.resize_height, self.resize_width), mode='constant')
            
			image_5 = io.imread(self.image_name_list[a - 5])
			image_5 = transform.resize(image_5, (self.resize_height, self.resize_width), mode='constant')
            
			image_6 = io.imread(self.image_name_list[a - 6])
			image_6 = transform.resize(image_6, (self.resize_height, self.resize_width), mode='constant')
            
			image_7 = io.imread(self.image_name_list[a - 7])
			image_7 = transform.resize(image_7, (self.resize_height, self.resize_width), mode='constant')
			'''
			image_0 = io.imread(self.image_name_list[idx])
			image_0 = transform.resize(image_0, (self.resize_height, self.resize_width), mode='constant')
			buffer[0, :, :, :] = image_0
			buffer[1, :, :, :] = image_0
			buffer[2, :, :, :] = image_0
			buffer[3, :, :, :] = image_0
			buffer[4, :, :, :] = image_0
			buffer[5, :, :, :] = image_0
			#buffer[6, :, :, :] = image_6            
			#buffer[7, :, :, :] = image_7           
            
		else :
			image_0 = io.imread(self.image_name_list[idx])
			image_0 = transform.resize(image_0, (self.resize_height, self.resize_width), mode='constant')

			image_1 = io.imread(self.image_name_list[idx - 1])
			image_1 = transform.resize(image_1, (self.resize_height, self.resize_width), mode='constant')

			image_2 = io.imread(self.image_name_list[idx - 2])
			image_2 = transform.resize(image_2, (self.resize_height, self.resize_width), mode='constant')

			image_3 = io.imread(self.image_name_list[idx - 3])
			image_3 = transform.resize(image_3, (self.resize_height, self.resize_width), mode='constant')

			image_4 = io.imread(self.image_name_list[idx - 4])
			image_4 = transform.resize(image_4, (self.resize_height, self.resize_width), mode='constant')
            
			image_5 = io.imread(self.image_name_list[idx - 5])
			image_5 = transform.resize(image_5, (self.resize_height, self.resize_width), mode='constant')
            
			#image_6 = io.imread(self.image_name_list[idx - 6])
			#image_6 = transform.resize(image_6, (self.resize_height, self.resize_width), mode='constant')
            
			#image_7 = io.imread(self.image_name_list[idx - 7])
			#image_7 = transform.resize(image_7, (self.resize_height, self.resize_width), mode='constant')

			buffer[0, :, :, :] = image_0
			buffer[1, :, :, :] = image_1
			buffer[2, :, :, :] = image_2
			buffer[3, :, :, :] = image_3
			buffer[4, :, :, :] = image_4
			buffer[5, :, :, :] = image_5
			#buffer[6, :, :, :] = image_6            
			#buffer[7, :, :, :] = image_7 

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

		if (3 == len(image.shape) and 2 == len(label.shape)):
			label = label[:, :, np.newaxis]
		elif (2 == len(image.shape) and 2 == len(label.shape)):
			image = image[:, :, np.newaxis]
			label = label[:, :, np.newaxis]

		label = transform.resize(label, (self.resize_height, self.resize_width), mode='constant', order=0,
								 preserve_range=True)

		sample = {'imidx': imidx, 'image': buffer, 'label': label}

		#sample = self.Random_Crop(sample)
		sample = self.ToTensorLab(sample)
		return sample

	def Random_Crop(self, sample):
		self.crop_size = (144, 144)
		imidx, buffer, label = sample['imidx'], sample['image'], sample['label']

		h, w = buffer.shape[1:3]
		new_h, new_w = self.crop_size

		top = np.random.randint(0, h - new_h)
		left = np.random.randint(0, w - new_w)
		buffer = buffer[:, top: top + new_h, left: left + new_w, :]
		label = label[top: top + new_h, left: left + new_w]
		return {'imidx': imidx, 'image': buffer, 'label': label}

	def ToTensorLab(self, sample):
		imidx, buffer, label = sample['imidx'], sample['image'], sample['label']
		tmpLbl = np.zeros(label.shape)

		if (np.max(label) < 1e-6):
			label = label
		else:
			label = label / np.max(label)
		# with rgb color
		tmpImg = np.zeros((buffer.shape[0], buffer.shape[1], buffer.shape[2], 3))
		#buffer = buffer / np.max(buffer)
		tmpImg = buffer
		tmpImg = tmpImg.transpose((3, 0, 1, 2))
		'''
		if buffer.shape[3] == 1:
			tmpImg[:, :, :, 0] = (buffer[:, :, :, 0] - 0.319) / 0.212
			tmpImg[:, :, :, 1] = (buffer[:, :, :, 0] - 0.319) / 0.212
			tmpImg[:, :, :, 2] = (buffer[:, :, :, 0] - 0.319) / 0.212
		else:
			tmpImg[:, :, :, 0] = (buffer[:, :, :, 0] - 0.319) / 0.212
			tmpImg[:, :, :, 1] = (buffer[:, :, :, 1] - 0.316) / 0.216
			tmpImg[:, :, :, 2] = (buffer[:, :, :, 2] - 0.302) / 0.228
		'''
		tmpImg = torch.from_numpy(tmpImg.copy())
		tmpImg = tmpImg.mul_(2.).sub_(255).div(255)

		tmpLbl[:, :, 0] = label[:, :, 0]

		# change the r,g,b to b,r,g from [0,255] to [0,1]
		# transforms.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225))
		#tmpImg = tmpImg.transpose((3, 0, 1, 2))
		tmpLbl = label.transpose((2, 0, 1))

		return {'imidx': torch.from_numpy(imidx), 'image': tmpImg,
				'label': torch.from_numpy(tmpLbl.copy())}


