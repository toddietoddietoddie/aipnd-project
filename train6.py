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

import copy
import time

import argparse
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print('after this...')
time.sleep(1)
print('Lets go do karate in the garage.')
time.sleep(.5)

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, help='Path to dataset ')
parser.add_argument('--gpu', default = False, type = bool, help='Use GPU if available')
#parser.add_argument('--gpu', default = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), help = 'Use GPU if available')
parser.add_argument('--epochs', type=int, default = 1, help='Number of epochs')
parser.add_argument('--arch', type=str, default = 'vgg19', help='Model architecture works with vgg or alexnet')
parser.add_argument('--learning_rate', type=float, default = 0.0001, help='Learning rate')
parser.add_argument('--hidden_units', type=int, default = 4000,help='Number of hidden units')
parser.add_argument('--save_dir', default = 'checkpoint.pth', type=str, help='Save trained model checkpoint to file')
parser.add_argument('--categories_json', action="store", default = 'cat_to_name.json', dest='categories_json', type=str, 
                    help='Path to file containing the categories.',
                        )



user_args, _ = parser.parse_known_args()

#define the number of epochs as a variable for later use
num_epochs = user_args.epochs

#define the network architecture as a variable for later use
architecture = user_args.arch

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
    print("vgg was my first love... Don't tell alexnet...-")
    print('converting over to work with vgg')

#adjust classifier features for alexnet
if architecture.startswith("alexnet"):
    input_size = alexnet_in_features
    print('-Rock on... Lets use Alexnet. Alex makes killer guacamole.-') 

model = models.__dict__[architecture](pretrained=True) 

#CREATE THE NEW CLASSIFIER FROM USER INPUT  

#user hidden_units - default = 4000
hidden_units = user_args.hidden_units
    
#New classifier for the flowers
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_size, hidden_units)),
    ('relu', nn.ReLU()),
    ('dropout1', nn.Dropout(p = .3)),
    ('fc2', nn.Linear(hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))

    
#change the classifier of the vgg19 model into our new classifier written above
model.classifier = classifier
model.to(device)

#define directory for path to dataset
directory = user_args.data_dir

#set criterion or loss function to NLLLoss that works well with a LogSoftmax output. 
#It also uses momentum which helps train faster.
criterion = nn.NLLLoss()

#learn_rate default = 0.0001
learn_rate = user_args.learning_rate
#the optimizer helps minimize the loss function and is thought of as the fastest - 
optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)

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

print('-Classifier input_features are now {} to work with {} network.-'.format(input_size, architecture))
print('-The hidden layer size is {}.-'.format(hidden_units))
    

# user loads categories_json file - default = "cat_to_name.json"
with open(user_args.categories_json, 'r') as f:
    cat_to_name = json.load(f) 
   

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

            
   
    gpu = user_args.gpu

    
    checkpoint_name = user_args.save_dir
        
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
                           'epoch': num_epochs,
                           'model_state_dict': model.state_dict(),
                           'classifier': model.classifier,
                           'optimizer': optimizer.state_dict(),
                           'class_to_idx': model.class_to_idx})
    
    print('Saving the checkpoint to {}'.format(checkpoint_name))
    torch.save(checkpoint, checkpoint_name)

            
            
            

            
######################################################################################## 

if __name__ == '__main__':
    train_model(model, criterion, optimizer, num_epochs, device = device)
#check accuracy of the network
#train_model(model, criterion, optimizer, num_epochs, device = device)
