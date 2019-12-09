import argparse
import os
import numpy as np
import json
from collections import OrderedDict
import utils

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


parser = argparse.ArgumentParser(description='Program that trains a neural network on a dataset of images')

parser.add_argument('data_directory', help="directory of the dataset", type = str)
parser.add_argument('--save_dir', help="directory to save checkpoint", default = os.getcwd(),type = str)

parser.add_argument('--architecture', help="select the model architecture", choices = ['resnet34', 'densenet121'], default = 'resnet34')
parser.add_argument('--learning_rate', help="learning rate of the algorithm", default = 0.03, type = float)
parser.add_argument('--hidden_units', help="number of nodes in the hidden layer", default = 250, type = int)
parser.add_argument('--epochs', help="number of epochs for training", default = 10, type = int)

parser.add_argument('--device', help="can either be GPU or CPU for model training", default = 'cuda', type = str)
args = parser.parse_args()



data_dir = args.data_directory
checkpoint_path = args.save_dir
checkpoint_path = checkpoint_path + '/checkpoint.pth'
learnrate = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
device = args.device
arc = args.architecture



##########################################################################################################
print('loading data...\n')

train_transforms = transforms.Compose([transforms.RandomRotation(25),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

#Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)


##########################################################################################################
print('training module using ',arc,'...')
if arc == 'resnet34':
    model = models.resnet34(pretrained=True)
    input_features = 512
elif arc == 'densenet121':
    model = models.densenet121(pretrained=True)
    input_features = 1024
    
    
for param in model.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_units, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

if arc == 'resnet34':
    model.fc = classifier
    optimizer = optim.SGD(model.fc.parameters(), lr=learnrate)
    
elif arc == 'densenet121':
    model.classifier = classifier
    optimizer = optim.SGD(model.classifier.parameters(), lr=learnrate)



if device == 'cuda':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == 'cpu':
        print('no cuda available\n')

print('device being used is: ',device)

model.to(device)
criterion = nn.NLLLoss()


#epochs initialized above

for e in range(epochs):
    model.train()
    running_loss = 0

    for images,labels in trainloader:
        images,labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()



    with torch.no_grad():
        model.eval()
        accuracy = 0
        test_loss = 0
        for images,labels in validloader:
            images,labels = images.to(device), labels.to(device)
            output = model.forward(images)
            test_loss += criterion(output, labels)
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
              "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
##########################################################################################################
print('\n\ntesting module using testing dataset...\n')
model.to(device)
with torch.no_grad():
    model.eval()
    accuracy = 0
    test_loss = 0
    criterion = nn.NLLLoss()
    for images,labels in testloader:
        images,labels = images.to(device), labels.to(device)
        output = model.forward(images)
        test_loss += criterion(output, labels)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim = 1)
        equals = top_class == labels.view(top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
       
    print("Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
          "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
    
##########################################################################################################
print('saving module to ', checkpoint_path,'...\n')
model.class_to_idx = train_data.class_to_idx

if arc == 'resnet34':
    classifier = model.fc
    
elif arc == 'densenet121':
    classifier = model.classifier
    
    
checkpoint = {'input_size': input_features,
              'output_size': 102,
              'hidden_layer': hidden_units,
              'class_to_idx':model.class_to_idx,
              'epochs':epochs,
              'optimizer': optimizer.state_dict(),
              'classifier': classifier,
              'architecture': arc,
              'state_dict': model.state_dict()}

torch.save(checkpoint, checkpoint_path)

##########################################################################################################
print('\n\n\nloading module from checkpoint...')
model = utils.load_checkpoint(checkpoint_path)
print('printing classifier to check if model loaded properly...\n\n')
if arc == 'resnet34':
    print(model.fc)
elif arc == 'densenet121':
    print(model.classifier)
    
print('training complete')