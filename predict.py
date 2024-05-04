import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import argparse
import json
from PIL import Image
import os, random
import numpy as np

def arguments():
    parser = argparse.ArgumentParser(description = 'Predict an image using classifier model.')

    parser.add_argument('--image_path', type = str, 
                        default = './flower_data/test/2/' + random.choice(os.listdir('./flower_data/test/2/')),                                               help = 'Path for the image')
    parser.add_argument('--load_checkpoint', type = str, default = 'checkpoint.pth', help = 'Path for the checkpoint')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Path to category names')
    parser.add_argument('--top_k', type = int, default = 5, help = 'Max k probabilities and names for the image')
    parser.add_argument('--gpu', action = 'store_true', help = 'Use of GPU')
    

    return parser.parse_args()

def print_command_lines(in_arg):
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\n     image_path =" ,in_arg.image_path,"\n     checkpoint_path =", in_arg.load_checkpoint, 
              "\n     category_names = ", in_arg.category_names,
              "\n     Device = ",'cuda' if in_arg.gpu else 'cpu')
        
def load_checkpoint(filename):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(filename, map_location = device)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
        
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
 
    img = Image.open(image)
    img.resize((256, 256))
    value = 0.5 * (256 - 224)
    img = img.crop((value, value, 256 - value, 256 - value))
    img = np.array(img)/255
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std

    return img.transpose(2,0,1)

def prediction(image_path, checkpoint_dir, catergory_names, topk, device):
    model = load_checkpoint(checkpoint_dir)

    with open(catergory_names, 'r') as f:
        cat_to_name = json.load(f)

    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    else:
        model.cpu()
    
    model.eval()
    image = process_image(image_path)
    
    image = torch.from_numpy(np.array([image])).float()
    
    image = Variable(image)
    if cuda:
        image = image.cuda()
        
    output = model.forward(image)
    
    probabilities = torch.exp(output).data

    prob = torch.topk(probabilities, topk)[0].tolist()[0] 
    index = torch.topk(probabilities, topk)[1].tolist()[0]
    
    ind = []
    for i in range(len(model.class_to_idx.items())):
        ind.append(list(model.class_to_idx.items())[i][0])

    
    label = []
    for i in range(topk):
        label.append(cat_to_name[ind[index[i]]])

    print("{:<20} {:<20}".format('Class', 'Probability'))
    for i in range(topk):
        print("{:<20} {:<20}".format(label[i], prob[i]))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
in_args = arguments()
print_command_lines(in_args)
prediction(in_args.image_path, in_args.load_checkpoint, in_args.category_names, in_args.top_k, device)