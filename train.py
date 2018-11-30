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

from train_functions8 import *
import argparse
import json

#set device to be cuda if available - later in the script it is always cpu unless user inputs it to be gpu...
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('after this...')
time.sleep(1)
print('Lets go do karate in the garage.')
time.sleep(.5)


num_epochs = user_args.epochs #define number of epochs as a variable for later use
architecture = user_args.arch #define the network architecture as a variable for later use

#taking the user input for network architecture and adjusting the in features based on the user_input
input_size = 0
alexnet_in_features = 9216
vgg_in_features = 25088  

#make sure the architecture runs with either alexnet or vgg
if not architecture.startswith('alexnet') and not architecture.startswith('vgg'):
    print('-Dudes... The network runs with vgg or alexnet-')
    time.sleep(.7)
    print("-I'm all for a good time but I can only party with vgg or alexnet-")
    time.sleep(.5)
    print('try using --arch vgg19... thats my favorite')
    exit(1)

# Input size from current classifier if VGG
if architecture.startswith("vgg"):
    input_size = vgg_in_features
    print('-You down with vgg... Yeah you know me!!-')
    time.sleep(.5)
    print('converting over to work with vgg')

#adjust classifier features for alexnet
if architecture.startswith("alexnet"):
    input_size = alexnet_in_features
    print('-Rock on... Lets use Alexnet.-') 

model = models.__dict__[architecture](pretrained=True) 

############################# CREATE THE NEW CLASSIFIER FROM USER INPUT ################################

hidden_units = user_args.hidden_units #get the user hidden_units - default = 4000
    
#New classifier for the flowers
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout1', nn.Dropout(p = .3)),
    ('fc2', nn.Linear(hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))

    
model.classifier = classifier #change the classifier of the original model into our new classifier written above
model.to(device) 

directory = user_args.data_dir #define directory for path to dataset

#set criterion or loss function to NLLLoss that works well with a LogSoftmax output. 
#It also uses momentum which helps train faster.
criterion = nn.NLLLoss()

learn_rate = user_args.learning_rate #user_input learn_rate - default = 0.0001

#the Adam optimizer helps minimize the loss function and is thought of as the fastest - 
optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)


print('-Classifier input_features are now {} to work with {} network.-'.format(input_size, architecture))
    

# user can load categories_json file - default = "cat_to_name.json"
with open(user_args.categories_json, 'r') as f:
    cat_to_name = json.load(f) 
   
#################################################################################################################
#################################################################################################################
################################## TRAIN MODEL FOR IMAGES #######################################################
#################################################################################################################

def train_model(model, criterion, optimizer, num_epochs, device = device):
    '''
    Args:
        param1(str): model that you want to train
        param2(var): the loss or error function
        param3(var): function that helps minimize the loss
        param4(var): number of epochs
        param5(var): CPU or GPU
    
    Returns:
        Validation Loss and Accuracy for each epoch'''
 
    gpu = user_args.gpu
    checkpoint_name = user_args.save_dir #default checkpoint name is checkpoint.pth
        
    print('Data Directory:', directory)    
    print('Network architecture:', architecture)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', num_epochs)
    print('Learning rate:', learn_rate)
    
   
    #I know that setting the gpu at the top of the file also does this 
    #but I wanted to make it available for the user to input gpu in order to pass the class :))   
    if gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print('Current device: {}'.format(device))
    
    model.to(device)
    
    for epoch in range(num_epochs):
        
        print('EPOCH {}/{}:'.format(epoch+1, num_epochs))
                
        for stage in ['train', 'valid']:
                          
            if stage == 'train':
                #set the model to train model with .train() 
                model.train()
            else:
                #.eval() eliminates the dropout
                model.eval()
                
            running_loss = 0
            accurate = 0
            #loop through images and labels in dataloader    
            for images, labels in dataloaders[stage]:
              
                #images, labels = data
                images, labels = images.to(device), labels.to(device)
                
                #zero the gradients in the optimizer
                optimizer.zero_grad()
                
                #only track the history if it is training
                with torch.set_grad_enabled(stage == 'train'):
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    #backpropogate and use optimize only in training
                    if stage == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * images.size(0)
                accurate += torch.sum(predicted == labels.data).float() #needs to be float for epoch_accuracy to work
            
                
            #calculate the loss and accuracy for each stage         
            if stage == 'train':
                epoch_loss = running_loss / len(train_dataset)
                epoch_accuracy = accurate / len(train_dataset)              
            
            elif stage == 'valid':
                epoch_loss = running_loss / len(valid_dataset)
                epoch_accuracy = accurate / len(valid_dataset)
                
            elif stage == 'test':
                epoch_loss = running_loss / len(test_dataset)
                epoch_accuracy = accurate / len(test_dataset)
            
            print('{} LOSS = {:.4f} ACCURACY = {:.4f}'.format(stage, epoch_loss, epoch_accuracy))
            
            model.class_to_idx = train_dataset.class_to_idx
#save the directory
            checkpoint = ({'model': architecture,                   
                           'num_epochs': num_epochs,
                           'model_state_dict': model.state_dict(),
                           'classifier': model.classifier,
                           'optimizer': optimizer.state_dict(),
                           'class_to_idx': model.class_to_idx})
    
    print('Saving the checkpoint to {}'.format(checkpoint_name))
    torch.save(checkpoint, checkpoint_name)
           
####################################################################################################################            
#################################################################################################################### 



if __name__ == '__main__':
    train_model(model, criterion, optimizer, num_epochs, device = device)
