# Imports here

#%matplotlib inline
#%config InlineBimport matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import OrderedDict
from PIL import Image

import torch
from torch import optim
import torch.utils.data as data
from torch import nn
import torch.nn.functional as F
from torch.autograd import variable
from torchvision import datasets, transforms, models

import copy
import time

import argparse

import json


print('after this...')
time.sleep(1)
print('Lets go do karate in the garage.')
time.sleep(.5)

#If cuda is available then we will use cuda.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, help='Path to dataset ')
parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
parser.add_argument('--epochs', type=int, default = 1, help='Number of epochs')
parser.add_argument('--arch', type=str, default = 'vgg19', help='Model architecture')
parser.add_argument('--learning_rate', type=float, default = 0.0001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default = 4000,help='Number of hidden units')
parser.add_argument('--checkpoint', type=str, help='Save trained model checkpoint to file')


user_args, _ = parser.parse_known_args()

#user_input arch... - default is vgg19

#Importing data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


directory = user_args.data_dir

#get a path to the correct folder
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
print(train_dir)
print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))

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

    

print(directory)   

architecture = user_args.arch    
model = models.__dict__[architecture](pretrained = True)

#model = models.vgg19(pretrained=True)

#set criterion or loss function to NLLLoss that works well with a LogSoftmax output. 
#It also uses momentum which helps train faster.
criterion = nn.NLLLoss()

learning_rate = user_args.learning_rate
#the optimizer helps minimize the loss function and is thought of as the fastest - 
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

num_epochs = user_args.epochs

#CREATE THE NEW CLASSIFIER FROM USER INPUT
input_size = 0
alexnet_in_features = 9216
vgg_in_features = 25088    

#user hidden_units - default = 4000
hidden_units = user_args.hidden_units

#works with both vgg and alexnet default is vgg19
pretrained_network = user_args.arch

if not pretrained_network.startswith('alexnet') and not pretrained_network.startswith('vgg'):
    print('-Dudes... The network runs with vgg or alexnet-')
    time.sleep(.7)
    print("-I'm all for a good time but I can only party with vgg or alexnet-")
    time.sleep(.5)
    print('try using --arch vgg19... thats my favorite')
    exit(1)

# Input size from current classifier if VGG
if pretrained_network.startswith("vgg"):
    input_size = vgg_in_features
    print('-vgg was my first love.-')
    time.sleep(.5)
    print("-Don't tell alexnet...-")

#adjust classifier features for alexnet
if pretrained_network.startswith("alexnet"):
    input_size = alexnet_in_features
    print('-Right on... Alex makes killer guacamole.-')  

#set gradient to false to freeze parameters
for param in model.parameters():
    param.requires_grad = False
     
#New classifier for the flowers
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout1', nn.Dropout(p = .3)),
    ('fc2', nn.Linear(hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))
    
model.classifier = classifier
print('-Classifier input_features are now {} to work with {} network.-'.format(input_size, architecture))
print('-The hidden layer size is {}.-'.format(hidden_units))

model.to(device)

######################################################################
###################### TRAIN MODEL FOR IMAGES ########################
######################################################################

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
    print(device)
     
    if user_args.arch:
        arch = user_args.arch     
        
    if user_args.hidden_units:
        hidden_units = user_args.hidden_units

    if user_args.epochs:
        num_epochs = user_args.epochs
            
    if user_args.learning_rate:
        learning_rate = user_args.learning_rate

    if user_args.gpu:
        gpu = user_args.gpu

    if user_args.checkpoint:
        checkpoint = user_args.checkpoint 
        
    print('Network architecture:', arch)
    print('Number of hidden units:', hidden_units)
    print('Number of epochs:', num_epochs)
    print('Learning rate:', learning_rate)

    dataloaders = {
        x: data.DataLoader(img_datasets[x], batch_size=4, shuffle=True, num_workers=2)
        for x in list(img_datasets.keys())
    }
 
    # Calculate dataset sizes.
    dataset_sizes = {
        x: len(dataloaders[x].dataset) 
        for x in list(img_datasets.keys())
    }
    
    num_labels = len(img_datasets['train'].classes)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        
        print('EPOCH {}/{}:'.format(epoch+1, num_epochs))
                
        for stage in ['train', 'valid']:
                          
            if stage == 'train':
                #set the model to train model with .train() 
                model.train()
            else:
                #.eval() eliminates the dropout
                model.eval()
                
            running_loss = 0.0
            accurate = 0
            #loop through images and labels in dataloader    
            for images, labels in dataloaders[stage]:             
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
        
        
      #  print('Done... Saving for future use...')

       # model.class_to_idx = train_dataset.class_to_idx
        #model_state = {
         #   'epoch': user_args.epochs,
          #  'state_dict': model.state_dict(),
           # 'optimizer_dict': optimizer.state_dict(),
            #'classifier': model.classifier,
            #'class_to_idx': model.class_to_idx,
            #'arch': user_args.arch
        #}

        #save_location = f'{}/{}.pth'.format(user_args.save_dir, user_args.save_name)
        #print("Saving checkpoint to {}".format(save_location)

        #torch.save(model_state, save_location)

            
########################################################################################   


train_model(model, criterion, optimizer, num_epochs, device = device)
