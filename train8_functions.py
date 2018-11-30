import numpy as np
from collections import OrderedDict
from PIL import Image

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.autograd import variable
from torchvision import datasets, transforms, models

import copy
import time

import argparse
import json

#set the device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#######################################################################################################################################
#######################################################################################################################################
############# Define the command line arguments below. The necassary argument is a saved checkpoint file ############################## 
#######################################################################################################################################
#######################################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to dataset ')
parser.add_argument('--gpu', default = False, type = bool, help='Use GPU if available True to use GPU - False to use CPU')
parser.add_argument('--epochs', type=int, default = 1, help='Number of epochs')
parser.add_argument('--arch', type=str, default = 'vgg19', help='Model architecture works with vgg or alexnet')
parser.add_argument('--learning_rate', type=float, default = 0.0001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default = 4000,help='Number of hidden units')
parser.add_argument('--save_dir', default = 'checkpoint.pth', type=str, help='Save trained model checkpoint to file')
parser.add_argument('--categories_json', action="store", default = 'cat_to_name.json', dest='categories_json', type=str, 
                    help='Path to file containing the categories.',
                        )

user_args, _ = parser.parse_known_args()

#######################################################################################################################################
#######################################################################################################################################

#define directory for path to dataset
directory = user_args.data_dir

#get a path to the correct folder using the directory
data_dir = 'flowers'
if directory == 'train_dir':
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
elif directory == 'valid_dir':
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
elif directory == 'test_dir':
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
#Importing data for predict function to work properly it must be imported another time
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#TRANSFORMS
# DONE: Define your transforms for the training, validation, and testing sets
train_data_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomRotation(20),
                                     transforms.RandomVerticalFlip(p = .5),
                                     transforms.RandomHorizontalFlip(p =.2),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])

valid_data_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

test_data_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

#datasets with ImageFolder
train_dataset = datasets.ImageFolder(train_dir, transform = train_data_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_data_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform = test_data_transforms)

#put the datasets into a list for easier indexing
img_datasets = {'train': train_dataset,
                'valid': valid_dataset,
                'test': test_dataset
               }

# DONE: Using the image datasets and the trainforms, define the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 70, shuffle = True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 40)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 30)

#put the dataloaders into  list
dataloaders = {'train': train_dataloader,
               'valid': valid_dataloader,
               'test': test_dataloader
              }


