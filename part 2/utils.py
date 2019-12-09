import argparse
import numpy as np
import json
from PIL import Image
import utils

import torch
from torchvision import models



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, 'cpu')
    if checkpoint['architecture'] == 'resnet34':        
        model = models.resnet34(pretrained=True)
        model.fc = checkpoint['classifier']
        
    elif checkpoint['architecture'] == 'densenet121':    
        model = models.densenet121(pretrained=True)
        model.classifier = checkpoint['classifier']
        
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = image.resize((256,256))
    width, height = image.size
    new_width, new_height = 224, 224
    #image.show()
    left = (width - new_width)/2
    upper = (height - new_height)/2
    right = (width + new_width)/2
    lower = (height + new_height)/2    
    
    image = image.crop((left, upper, right, lower))
    #image.show()
    np_image = np.array(image)
    np_image = np_image/255   
    #print(np_image.shape)
    
    for x in range(224):
        for y in range(224):
            for i in range(3):

                if i == 0:
                    np_image[x,y,i] = (np_image[x,y,i] - 0.485)/0.229
                    
                elif i == 1:
                    np_image[x,y,i] = (np_image[x,y,i] - 0.456)/0.224
                    
                else:
                    np_image[x,y,i] = (np_image[x,y,i] - 0.406)/0.225
            
            

    np_image = np_image.transpose((2,0,1))
    #print(np_image.shape)
    return torch.tensor(np_image)



def predict(image_path, model_checkpoint_path, topk, category_names, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    image = process_image(image)

    

    
    #have to add a dimension cuz my model expects a list of images not a single image
    image = torch.unsqueeze(image, 0).type(torch.FloatTensor)
    #print(image_tensor.shape)
    model = load_checkpoint(model_checkpoint_path)
    class_to_idx = model.class_to_idx
    '''
    ps.topk for some reason returns the keys instead of the values we need to search in cat_to_name to find the names, 
    so we have to invert the keys and values, call the values then search in cat_to_name. OR we can just invert the dict 
    while saving the checkpoint
    '''
    class_to_idx_inv = {v: k for k, v in class_to_idx.items()}
    
    device = 'cpu'
    if(gpu):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == 'cpu':
            print('no cuda available\n')
            
    print('device being used is: ',device,'\n\n')
    model.to(device)
    image = image.to(device)
    model.eval()

    output = model.forward(image)    
    ps = torch.exp(output)
    top_p, top_classes = ps.topk(topk, dim = 1)
    ####print('top classes of 1 img',top_classes[0])
    
    probabilities = [round(i.item(),2) for i in top_p[0]]
    labels = [class_to_idx_inv[i.item()] for i in top_classes[0]]
    ####print('labels numbers: ',labels)
    print('class numbers are: ', labels)
    if category_names != 'None':
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)             
            labels = [cat_to_name[i] for i in labels]
            
    ####print('labels names: ', labels)
    return probabilities, labels