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

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#######################################################################
############ LOAD WORKING MODEL OR CHECKPOINT #########################
#######################################################################

def load_checkpoint(filepath):
    '''Args:
        Param1(file): checkpoint name as readline file
       Returns:
        The pretrained model for image classification of flowers'''
    
    #cant name variable checkpoint. confuses with checkpoint list in previous cell
    chckpnt = torch.load(filepath)
    
    if chckpnt['model'] == 'vgg19':
        model = models.vgg19(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
    else:
        print('program built to work with vgg19')
        
        
    model.class_to_idx = chckpnt['class_to_idx']
    
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, 4000)),
        ('relu', nn.ReLU()),
        ('dropout1', nn.Dropout(p = .3)),
        ('fc2', nn.Linear(4000, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    
    model.classifier = classifier
    
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
    
    #np_image = np.array(image_path, dtype = 'float32')
    #np_image = (np_image / 255.0)
    
    np_image = (image_path - mean) / std
    
    np_image = np_image.transpose((2,0,1))
    
    return np_image
    
    
##############################################################################################    
    
############################################################################################
#################### RETURN ORIGINAL IMAGE AFTER PROCESS_IMAGE() ###########################
############################################################################################

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    if title:
        plt.title(title)
        
        
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

###########################################################################################

#########################################################################################
################## PREDICT THE TOP 5 CLASS PROBABILITIES ################################
#########################################################################################

def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''   
    #run the predict function through cpu
    model = model.to('cpu')
    #process image and make correct type
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    
    
    #predict the top 5 #unsqueeze the image_tensor adds dimension of size 1 at specified dimension(0)
    probs = torch.exp(model.forward(image_tensor.unsqueeze_(0)))

    
    top_probs, top_labs = probs.topk(5)
    top_probs = top_probs.detach().numpy().tolist()[0]
    top_labs = top_labs.detach().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    best_labels = [idx_to_class[label] for label in top_labs]
    best_flowers = [cat_to_name[idx_to_class[label]] for label in top_labs]
    
    return top_probs, best_labels, best_flowers
        
#########################################################################################

#########################################################################################
############### Double check the network ################################################
#########################################################################################
def sanity_check(image_path, model):
    
    plt.figure(figsize = (7,11))
    ax = plt.subplot(2,1,1)
    
    flower_num = image_path.split('/')[2]
    flower_title = cat_to_name[flower_num]
    
    image = process_image(image_path)
    imshow(image, ax, flower_title) 

    top_probs, best_labels, best_flowers = predict(image_path, model)
    
    plt.subplot(2,1,2)
    sns.barplot(x=best_flowers, y=top_probs, color= sns.color_palette()[9])
    plt.xticks(rotation = 45)
    plt.ylabel('percent of confidence')
    plt.xlabel('top 5 flower options')
    plt.show()

###################################################################################


