#first we rock then we roll

from collections import OrderedDict
from PIL import Image

import numpy as np
import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.autograd import variable
from torchvision import datasets, transforms, models

import argparse

import json

#######################################################################################################################################
#######################################################################################################################################
############# Define the command line arguments below. The necassary argument is a saved checkpoint file ############################## 
#######################################################################################################################################
#######################################################################################################################################
parser = argparse.ArgumentParser()
parser.add_argument('checkpoint_path', action = 'store', help = 'path to checkpoint file')
parser.add_argument('--path_to_image', default = './flowers/test/101/image_07988.jpg', action = 'store', help = 'path to image file')
parser.add_argument('--save_dir', action = 'store', default = '.', dest = 'save_dir', type = str, help = 'Directory to save training checkpoint')
parser.add_argument('--top_k', action = 'store', default = 5, dest = 'top_k', type = int, help = 'Returns top k most likely classes.')
parser.add_argument('--gpu', default = False, type = bool, help='Use GPU if available')
parser.add_argument('--categories_json', action = 'store', default = 'cat_to_name.json', dest = 'categories_json', type = str,
                    help = 'file containing categories',
                   )                                                                                                         
                 
user_args, _ = parser.parse_known_args()

#######################################################################################################################################
#######################################################################################################################################

category_names = user_args.categories_json
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)
    
#Importing data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


#######################################################################
############ LOAD WORKING MODEL OR CHECKPOINT #########################
#######################################################################

def load_checkpoint(filepath):
    '''Args:
        Param1(file): checkpoint name as readline file
       Returns:
        The pretrained model for image classification of flowers'''
    
    
    #cant name variable checkpoint. confuses with checkpoint list 
    chckpnt = torch.load(filepath)
    saved_model = chckpnt['model']           
    saved_classifier = chckpnt['classifier']

    
    if saved_model.startswith('vgg') or saved_model.startswith('alexnet'):
        model = models.__dict__[saved_model](pretrained = True)
        #model = models.vgg19(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print('program built to work with vgg19 or alexnet')
        
        
    model.class_to_idx = chckpnt['class_to_idx']
        
    model.classifier = saved_classifier
    
    model.load_state_dict(chckpnt['model_state_dict'])
    
    return model

##################################################################
    


######################################################################################
################# PROCESS PIL IMAGE ##################################################
######################################################################################

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # DONE: Process a PIL image for use in a PyTorch model
    
    ##adjust the size to smaller side being 256 and larger size to scale of 256
    image_path = Image.open(image_path)
    image_size = image_path.size
    
    if image_size[0] < image_size[1]:
        image_newsize = (int(256), int((256 * image_size[1]/image_size[0])))   
        image_path.thumbnail(image_newsize)
                  
    else:
        image_newsize = (int((256 * image_size[0]/image_size[1])), int(256))
        image_path.thumbnail(image_newsize)
    
    ##center crop to 224##
    left_margin = (image_newsize[0] - 224)/2
    bottom_margin = (image_newsize[1] - 224)/2 
    top_margin = bottom_margin + 224
    right_margin = left_margin + 224

   
    
    crop_out_size = (left_margin, bottom_margin, right_margin, top_margin)
        
    image_path = image_path.crop((left_margin, bottom_margin, right_margin, top_margin))

    image_path = np.array(image_path)/255
    
    mean = np.array([0.485, 0.456, 0.406], dtype = 'float32')
    std = np.array([0.229, 0.224, 0.225], dtype = 'float32')
    
    
    np_image = (image_path - mean) / std
    
    np_image = np_image.transpose((2,0,1))
    
    return np_image
    
    
############################################################################################## 

