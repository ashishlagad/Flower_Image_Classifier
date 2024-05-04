import torch
import torchvision
from torch import nn
from torch import optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import argparse
import json

def arguments():
    parser = argparse.ArgumentParser(description = 'Training an image classifier model for a given dataset')
    
    parser.add_argument('--data_dir', default = 'flower_data', help = 'Path of the dataset') 
    parser.add_argument('--save_dir', type = str, default = 'checkpoint.pth', help = 'Path for saving directory')
    parser.add_argument('--arch', type = str, default = 'densenet121', help = 'Model architecture to be used')
    parser.add_argument('--learning_rate', type = float, default = 0.003, help = 'Learning rate for optimizer')
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'Number of hidden units to be used')
    parser.add_argument('--epochs', type = int, default = 1, help = 'Number of epochs')
    parser.add_argument('--gpu',action = 'store_true', help = 'Use of GPU for training.')

    return parser.parse_args()

def print_command_lines(in_arg):
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\n     data_dir = ",in_arg.data_dir,"\n     save_dir =", in_arg.save_dir, 
              "\n     arch = ", in_arg.arch, "\n     learning_rate = ", in_arg.learning_rate,
              "\n     hidden_Units = ", in_arg.hidden_units, "\n     epochs = ",in_arg.epochs,
              "\n     Device =",'cuda' if in_arg.gpu else 'cpu')
       
        
def train(arch, data_directory, save_dir, epochs, learn_rate, hidden_units, device):
    
    densenet_weights = torchvision.models.DenseNet121_Weights.DEFAULT
    vgg_weights = torchvision.models.VGG16_Weights.DEFAULT
    densenet121 = models.densenet121(densenet_weights)
    vgg16 = models.vgg16(vgg_weights)

    models_list = {'densenet121': densenet121, 'vgg16': vgg16}
    
    train_data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                        ])

    test_data_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])
                                        ])
    data_dir = data_directory
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    
    train_data = datasets.ImageFolder(train_dir, transform = train_data_transforms)
    validate_data = datasets.ImageFolder(valid_dir, transform = test_data_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    validateloader = torch.utils.data.DataLoader(validate_data, batch_size = 64)

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    class_to_idx = dict()
    for i in cat_to_name:
        class_to_idx[cat_to_name[i]] = int(i)

    model = models_list[arch]

    for each in model.parameters():
        each.require_grad = False
    
    if(arch == 'densenet121'):
        classifier = nn.Sequential(nn.Linear(1024, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(p = 0.2),
                            nn.Linear(hidden_units, int(hidden_units/2)),
                            nn.ReLU(),
                            nn.Dropout(p = 0.2),
                            nn.Linear(int(hidden_units/2), 102),
                            nn.LogSoftmax(dim = 1))
    else:
        classifier = nn.Sequential(nn.Linear(25088, 1024),
                            nn.ReLU(),
                            nn.Dropout(p = 0.2),
                            nn.Linear(1024, hidden_units),
                            nn.ReLU(),
                            nn.Dropout(p = 0.2),
                            nn.Linear(hidden_units, int(hidden_units/2)),
                            nn.ReLU(),
                            nn.Dropout(p = 0.2),
                            nn.Linear(int(hidden_units/2), 102),
                            nn.LogSoftmax(dim = 1))
    

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr = learn_rate)
    
    model.to(device)
    train_loss = 0

    print("....................................Training.................................... \n")
    cuda = torch.cuda.is_available()
    if cuda:
        model.cuda()
    else:
        model.cpu()
        
    train_loss = 0

    for epoch in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
        
            optimizer.zero_grad()
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
        
        else:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in validateloader:
                    images, labels = images.to(device), labels.to(device)
                    output = model.forward(images)
                    batch_loss = criterion(output, labels)
                
                    valid_loss += batch_loss.item()
                
                    #Calculate accuracy
                    ps = torch.exp(output)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
            print(f"Epoch {epoch+1}/{epochs}..\n "
                  f"Train loss: {train_loss/len(trainloader):.3f}.. "
                  f"Validation loss: {valid_loss/len(validateloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validateloader):.3f}")
            train_loss = 0
            model.train()

    model.class_to_idx = train_data.class_to_idx

    checkpoint = {'arch': arch,
                'learning_rate': 0.003,
                'batch_size': 64,
                'classifier' : classifier,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, save_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
in_args = arguments()
print_command_lines(in_args)
train(in_args.arch, in_args.data_dir, in_args.save_dir, in_args.epochs, in_args.learning_rate, in_args.hidden_units, device)