import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import OrderedDict
from PIL import Image

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.autograd import variable
from torchvision import datasets, transforms, models

import argparse
#from model_functions import train_model, check_accuracy
from utility_functions import load_checkpoint, process_image, imshow, predict, sanity_check

from train6 import *

import json


# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path', action = 'store', help = 'path to checkpoint file')
parser.add_argument('path_to_image', action = 'store', help = 'path to image file')
parser.add_argument('--save_dir', action = 'store', default = '.', dest = 'save_dir', type = str, help = 'Directory to save training checkpoint')
parser.add_argument('--top_k', action = 'store', default = 5, dest = 'top_k', type = int, help = 'Returns top k most likely classes.')
parser.add_argument('--category_names', action = 'store', default = 'cat_to_name.json', dest = 'categories_json', type = str,
                    help = 'file containing categories',
                   )                                                                                                         
parser.add_argument('--gpu', action = 'store_true',  dest = 'use_gpu', default = False, help = 'use gpu')
                 

user_args, _ = parser.parse_known_args()

load_checkpoint(user_args.checkpoint_path)

#Importing data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#image_path = (test_dir + '/1/image_06743.jpg')


image_path = user_args.path_to_image
prediction = predict(image_path, model)
print('prediction 2')
print(prediction)
