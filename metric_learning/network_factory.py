import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import models
import statistics
import time
import copy
from tqdm import tqdm
from collections import OrderedDict





class SwAVResnet (nn.Module):
    def __init__(self, num_classes):
        super(SwAVResnet, self).__init__()
        self.aerialresnet = torch.hub.load('facebookresearch/swav', 'resnet50')
        self.groundresnet = torch.hub.load('facebookresearch/swav', 'resnet50')
        self.num_classes = num_classes

    def forward (self, aerial, ground):
    	#Aerial encoder
        x_a = self.aerialresnet.conv1(aerial)
        x_a = self.aerialresnet.bn1(x_a)
        x_a = self.aerialresnet.relu(x_a)
        x_a = self.aerialresnet.maxpool(x_a)
        x_a = self.aerialresnet.layer1(x_a)
        x_a = self.aerialresnet.layer2(x_a)
        x_a = self.aerialresnet.layer3(x_a)
        x_a = self.aerialresnet.layer4(x_a)
        x_a = self.aerialresnet.avgpool(x_a)
        
        #Ground encoder
        x_g = self.groundresnet.conv1(ground)
        x_g = self.groundresnet.bn1(x_g)
        x_g = self.groundresnet.relu(x_g)
        x_g = self.groundresnet.maxpool(x_g)
        x_g = self.groundresnet.layer1(x_g)
        x_g = self.groundresnet.layer2(x_g)
        x_g = self.groundresnet.layer3(x_g)
        x_g = self.groundresnet.layer4(x_g)
        x_g = self.groundresnet.avgpool(x_g)

        #Normalizing features
        norm_a = torch.norm(x_a, p='fro', dim=1).detach()
        for i in range(x_a.shape[0]):
        	x_a[i, :] = x_a[i, :]/norm_a[i]
        x_a = torch.squeeze(x_a)

        norm_g = torch.norm(x_g, p='fro', dim=1).detach()
        for i in range(x_a.shape[0]):
        	x_g[i, :] = x_g[i, :]/norm_g[i]
        x_g = torch.squeeze(x_g)

        distance_matrix = 2 - 2 * torch.matmul(x_g, torch.transpose(x_a, 0, 1))

        return distance_matrix, x_a, x_g



def infer (model, aerial, ground):
    aerial = aerial.cuda()
    ground = ground.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    with torch.set_grad_enabled(False):
        dist_matrix = model(aerial, ground)        