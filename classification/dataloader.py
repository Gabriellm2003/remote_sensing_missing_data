from torchvision import datasets, transforms
import os
import torch
from skimage.io import imread
from skimage.transform import resize
import skimage.transform as scikit_transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]#return image path

def create_dataloader(data_dir, input_size, batch_size, mean, std):
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'validation': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    }

    image_datasets = {x: MyImageFolder(os.path.join(data_dir, x), 
    	              data_transforms[x]) for x in ['train', 'validation']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                        batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True) 
                        for x in ['train', 'validation']}
    return dataloaders_dict


def create_dataloader_sentinel(data_dir, input_size, batch_size):    
    image_datasets = {x: MyImageFolder(os.path.join(
            data_dir, x), transform = transforms.ToTensor(), loader=mySentinelLoader) for x in ['train', 'validation']}
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                        batch_size=batch_size, shuffle=True, num_workers=8) 
                        for x in ['train', 'validation']}
    return dataloaders_dict


def mySentinelLoader (img_name):
    #mean = [0.16, 0.136, 0.123, 0.115, 0.128, 0.175, 0.199, 0.189, 0.212, 0.059,0.002, 0.158, 0.108]
    mean = [0.153, 0.129, 0.119, 0.114, 0.135, 0.206, 0.239, 0.229, 0.259, 0.071, 0.002, 0.199, 0.132]
    #std =  [0.141, 0.126, 0.117, 0.116, 0.123, 0.156, 0.173, 0.166, 0.182, 0.064, 0.004, 0.149, 0.11]
    std = [0.132, 0.116, 0.109, 0.109, 0.122, 0.168, 0.187, 0.182, 0.199, 0.071, 0.002, 0.167, 0.124]
    image = imread(img_name)[:,:,:]
    image = scikit_transform.resize(image, (224,224)).astype(image.dtype)
    for i in range(13):
        image [:, :, i] = (image[:,:,i]-mean[i])/std[i] 
    return image

