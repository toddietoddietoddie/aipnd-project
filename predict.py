from collections import OrderedDict
from PIL import Image

import torch
from torch import optim
from torch import nn
import torch.nn.functional as F
from torch.autograd import variable
from torchvision import datasets, transforms, models

import argparse

from predict_args_functions import *


import json

#command line arguments are in the predict_args_functions.py file. 
#The load_checkpoint and process_image functions needed to run the predict function below are also in the predict_args_functions.py


#########################################################################################
################## PREDICT THE TOP CLASS PROBABILITIES ################################
#########################################################################################

def predict(image_path, model):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #run the predict function through cpu or gpu. If user requests gpu then we use gpu. Default is Cpu
    
    user_gpu_request = user_args.gpu
    if user_gpu_request and torch.cuda.is_available():
        device = torch.device("cuda:0")
        #adjust tensor_type for the image_tensor to work properly with gpu or cpu
        tensor_type = torch.cuda.FloatTensor
    else:
        device = torch.device("cpu")
        #adjust tensor_type for the image_tensor to work properly with gpu or cpu
        tensor_type = torch.FloatTensor
    print('Dudes... Our current device is: {}'.format(device))
    
    model.to(device)
    
    
  

    #process image and make correct type
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(tensor_type)
    
    
    #predict the top 5 #unsqueeze the image_tensor adds dimension of size 1 at specified dimension(0)
    probs = torch.exp(model.forward(image_tensor.unsqueeze_(0)))
    
    #User topk integer request for a numerical amount of likely classes - default is 5 or top 5 classes will display
    user_topk_request = user_args.top_k
    
    top_probs, top_labs = probs.topk(user_topk_request)
   
    #received error message stating that can't convert CUDA tensor to numpy and to Use tensor.cpu() to copy tensor to host memory first.
    top_probs = top_probs.detach().cpu().numpy().tolist()[0]
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    best_labels = [idx_to_class[label] for label in top_labs]
    best_flowers = [cat_to_name[idx_to_class[label]] for label in top_labs]
    
    return top_probs, best_labels, best_flowers
    
    
#########################################################################################



    
model = load_checkpoint(user_args.checkpoint_path)

image_path = user_args.path_to_image

prediction = predict(image_path, model)
print(prediction)
