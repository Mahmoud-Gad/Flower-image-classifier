import argparse
import numpy as np
import json
from PIL import Image
import utils

import torch
from torchvision import models


parser = argparse.ArgumentParser(description='Program that predicts the class of an image')

parser.add_argument('image_path', help="The path of the image", type = str)
parser.add_argument('checkpoint', help="The path to load the model from a checkpoint", type = str)
parser.add_argument('--top_k', help="number of top classes to show", default = 1, type = int)
parser.add_argument('--category_names', help="use to map classes' numbers to names", default = 'None', type = str)

parser.add_argument('--gpu', action = 'store_true',help="can either be GPU or CPU for model training", default = False)
args = parser.parse_args()



image_path = args.image_path
checkpoint = args.checkpoint
topk = args.top_k
category_names = args.category_names
gpu = args.gpu


print('\n\n\nloading module from checkpoint...')
model = utils.load_checkpoint(checkpoint)


probs, classes = utils.predict(image_path, checkpoint, topk, category_names, gpu)
print('probabilities are: ', probs)
print('classes labels are: ', classes)