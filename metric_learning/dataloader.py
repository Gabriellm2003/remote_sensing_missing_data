import numpy as np
from skimage.io import imread
import os
import random
from skimage.transform import resize, rescale
from PIL import ImageFile
from torch.utils.data import TensorDataset, DataLoader
import torch
from torchvision import transforms
from PIL import Image
ImageFile.LOAD_TRUNCATED_IMAGES = True


random.seed(30) #setting seed for reproducibility



def normalize(img):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for i in range(3):
        img [:, :, i] = (img[:,:,i]-mean[i])/std[i]
    return img 


class DataLoader:
    def __init__(self, polar_path, ground_path, batch):
        self.polar_root = polar_path
        self.ground_root = ground_path
        self.train_img_polar = {}
        self.train_img_ground = {}
        self.val_img_polar = []
        self.val_img_ground = []
        self.batch_size = 0
        self.train_data_size = 0
        self.__cur_id = 0
        self.__cur_val_id = 0
        self.__cur_test_id = 0
        self.test_img_polar = []
        self.test_img_ground = []
        self.transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

        for c in os.listdir(os.path.join(self.polar_root, 'train')):
            if c not in self.train_img_polar.keys():
                self.train_img_polar[c] = []
            for img in os.listdir(os.path.join(self.polar_root, 'train', c)):
                if (c == 'apartment'):
                    for t in range(3):
                        self.train_img_polar[c].append(os.path.join(self.polar_root, 'train', c, img))
                elif (c == 'house'):
                    for t in range(2):
                        self.train_img_polar[c].append(os.path.join(self.polar_root, 'train', c, img))
                elif (c == 'industrial'):
                    for t in range(5):
                        self.train_img_polar[c].append(os.path.join(self.polar_root, 'train', c, img))
                elif (c == 'parking_lot'):
                    for t in range(11):
                        self.train_img_polar[c].append(os.path.join(self.polar_root, 'train', c, img))
                elif (c == 'religious'):
                    for t in range(9):
                        self.train_img_polar[c].append(os.path.join(self.polar_root, 'train', c, img))
                elif (c == 'school'):
                    for t in range(8):
                        self.train_img_polar[c].append(os.path.join(self.polar_root, 'train', c, img))
                elif (c == 'store'):
                    for t in range(9):
                        self.train_img_polar[c].append(os.path.join(self.polar_root, 'train', c, img))
                else:
                    self.train_img_polar[c].append(os.path.join(self.polar_root, 'train', c, img))




        for c in os.listdir(os.path.join(self.ground_root, 'train')):
            if c not in self.train_img_ground.keys():
                self.train_img_ground[c] = []
            for img in os.listdir(os.path.join(self.ground_root, 'train', c)):
                if (c == 'apartment'):
                    for t in range(3):
                        self.train_img_ground[c].append(os.path.join(self.ground_root, 'train', c, img))
                elif (c == 'house'):
                    for t in range(2):
                        self.train_img_ground[c].append(os.path.join(self.ground_root, 'train', c, img))
                elif (c == 'industrial'):
                    for t in range(5):
                        self.train_img_ground[c].append(os.path.join(self.ground_root, 'train', c, img))
                elif (c == 'parking_lot'):
                    for t in range(11):
                        self.train_img_ground[c].append(os.path.join(self.ground_root, 'train', c, img))
                elif (c == 'religious'):
                    for t in range(9):
                        self.train_img_ground[c].append(os.path.join(self.ground_root, 'train', c, img))
                elif (c == 'school'):
                    for t in range(8):
                        self.train_img_ground[c].append(os.path.join(self.ground_root, 'train', c, img))
                elif (c == 'store'):
                    for t in range(9):
                        self.train_img_ground[c].append(os.path.join(self.ground_root, 'train', c, img))
                else:
                    self.train_img_ground[c].append(os.path.join(self.ground_root, 'train', c, img))


        self.lowest_samples = 999999
        for k in self.train_img_polar.keys():
            #self.train_data_size += len(self.train_img_polar[k])
            if (len(self.train_img_polar[k]) < self.lowest_samples):
                self.lowest_samples = len(self.train_img_polar[k])
        self.train_data_size = len(self.train_img_polar.keys()) * self.lowest_samples
        self.batch_size = len(self.train_img_polar.keys())

        for c in os.listdir(os.path.join(self.polar_root, 'val')):
            for img in os.listdir(os.path.join(self.polar_root, 'val', c)):
                self.val_img_polar.append(os.path.join(self.polar_root, 'val', c, img))

        for c in os.listdir(os.path.join(self.ground_root, 'val')):
            for img in os.listdir(os.path.join(self.ground_root, 'val', c)):
                self.val_img_ground.append(os.path.join(self.ground_root, 'val', c, img))


        self.val_data_size = len(self.val_img_ground)

        for c in os.listdir(os.path.join(self.polar_root, 'test')):
            for img in os.listdir(os.path.join(self.polar_root, 'test', c)):
                self.test_img_polar.append(os.path.join(self.polar_root, 'test', c, img))

        for c in os.listdir(os.path.join(self.ground_root, 'test')):
            for img in os.listdir(os.path.join(self.ground_root, 'test', c)):
                self.test_img_ground.append(os.path.join(self.ground_root, 'test', c, img))


        
        self.test_data_size = len(self.test_img_polar)
        #print (len(self.val_img_polar))
        #print (len(self.val_img_ground))

    def next_train_batch_scan(self, batch):
        #print (self.lowest_samples)

        if (self.__cur_id == 0):
            for i in range(10):
                for k in self.train_img_polar.keys():
                    random.shuffle(self.train_img_polar[k])
                    random.shuffle(self.train_img_ground[k])
        
        if (self.__cur_id + 2 >= self.lowest_samples):
            self.__cur_id = 0
            return None, None

        batch_sat = torch.zeros(self.batch_size, 3, 224, 224, dtype=torch.float)
        batch_grd = torch.zeros(self.batch_size, 3, 224, 224, dtype=torch.float)
        key_list = self.train_img_polar.keys()

        for x, k in enumerate(key_list):
            img_idx = self.__cur_id
            img = Image.open(self.train_img_polar[k][img_idx]).convert('RGB')
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            batch_sat[x, :, :, :] = img[:,0:3,:,:]

            img = Image.open(self.train_img_ground[k][img_idx]).convert('RGB')
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            batch_grd[x, :, :, :] = img[:,0:3,:,:]
        
        self.__cur_id += 1
        return batch_sat, batch_grd


    def next_val_batch_scan (self, batch_size):

        #print ('ID:')
        #print (__cur_val_id)
        if (self.__cur_val_id >= self.val_data_size):
            print('li tudo.')
            self.__cur_val_id = 0
            return None, None, None, None
        elif (self.__cur_val_id + batch_size >= self.val_data_size):
            batch_size = self.val_data_size - self.__cur_val_id


        batch_sat = torch.zeros(batch_size, 3, 224, 224, dtype=torch.float)
        batch_grd = torch.zeros(batch_size, 3, 224, 224, dtype=torch.float)
        batch_ids_sat = np.empty([batch_size], dtype="S100")
        batch_ids_grd = np.empty([batch_size], dtype="S100")

        for i in range(batch_size):
            img_idx = self.__cur_val_id + i
            img = Image.open(self.val_img_polar[img_idx]).convert('RGB')
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            batch_sat[i, :, :, :] = img[:,0:3,:,:]

            img = Image.open(self.val_img_ground[img_idx]).convert('RGB')
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            batch_grd[i, :, :, :] = img[:,0:3,:,:]
            
            batch_ids_sat[i] = self.val_img_polar[img_idx].split('/')[-2] + '___' + self.val_img_polar[img_idx].split('/')[-1].replace('.png', '')
            batch_ids_grd[i] = self.val_img_ground[img_idx].split('/')[-2] + '___' + self.val_img_ground[img_idx].split('/')[-1].replace('.png', '')
            #batch_ids.append(self.train_img_list[img_id][2])
            #batch_idx += 1

        self.__cur_val_id += batch_size

        return batch_sat, batch_grd, batch_ids_sat, batch_ids_grd


    def next_test_batch_scan (self, batch_size):

        if (self.__cur_test_id >= self.test_data_size):
            print('li tudo.')
            self.__cur_test_id = 0
            return None, None, None, None
        elif (self.__cur_test_id + batch_size >= self.test_data_size):
            batch_size = self.test_data_size - self.__cur_test_id
        elif (batch_size == 0):
            print('li tudo.')
            return None, None, None, None

        #print (batch_size)
        #print (self.__cur_test_id)
        batch_sat = torch.zeros(batch_size, 3, 224, 224, dtype=torch.float)
        batch_grd = torch.zeros(batch_size, 3, 224, 224, dtype=torch.float)
        batch_ids_sat = np.empty([batch_size], dtype="S100")
        batch_ids_grd = np.empty([batch_size], dtype="S100")

        for i in range(batch_size):
            img_idx = self.__cur_test_id + i

            #satelite
            img = Image.open(self.test_img_polar[img_idx]).convert('RGB')
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            batch_sat[i, :, :, :] = img[:,0:3,:,:]

            img = Image.open(self.test_img_ground[img_idx]).convert('RGB')
            img = self.transform(img)
            img = torch.unsqueeze(img, 0)
            batch_grd[i, :, :, :] = img[:,0:3,:,:]
            
            batch_ids_sat[i] = self.test_img_polar[img_idx].split('/')[-2] + '___' + self.test_img_polar[img_idx].split('/')[-1].replace('.png', '')
            batch_ids_grd[i] = self.test_img_ground[img_idx].split('/')[-2] + '___' + self.test_img_ground[img_idx].split('/')[-1].replace('.png', '')
            #batch_ids.append(self.train_img_list[img_id][2])
            #batch_idx += 1

        self.__cur_test_id += batch_size

        return batch_sat, batch_grd, batch_ids_sat, batch_ids_grd

    def get_val_dataset_size(self):
        return self.val_data_size

    def get_test_dataset_size(self):
        return self.test_data_size

    def reset_scan(self):
        self.__cur_val_id = 0