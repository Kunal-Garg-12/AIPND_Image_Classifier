# Imports here
import torch
from torchvision import transforms, datasets, models
from torch import nn, optim
import torchvision.models as models
from collections import OrderedDict
import json
import argparse
    
#Take arguments from CLI
parser = argparse.ArgumentParser()

parser.add_argument('data_dir', type=str, default="./flowers/")
parser.add_argument('--save_dir', type=str, default='./checkpoint.pth')
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--learning_rate', type=float, default=.001)
parser.add_argument('--hidden_layer', type=int, default=1024)
parser.add_argument('--epochs', type=int, default=12)
parser.add_argument('--gpu', default='gpu', type=str)

inputs = parser.parse_args()
data_dir = inputs.data_dir
save_dir = inputs.save_dir
arch = inputs.arch
lr = inputs.learning_rate
hidden_layer = inputs.hidden_layer
gpu = inputs.gpu
epochs = inputs.epochs

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
train_data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                               transforms.RandomHorizontalFlip(p=0.4),
                                               transforms.RandomVerticalFlip(p=0.4),
                                               transforms.RandomRotation(60),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

valid_data_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_image_datasets = datasets.ImageFolder(train_dir, transform=train_data_transforms)
test_image_datasets = datasets.ImageFolder(test_dir, transform=valid_data_transforms)
valid_image_datasets = datasets.ImageFolder(valid_dir, transform=valid_data_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_dataloader = torch.utils.data.DataLoader(train_image_datasets, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_image_datasets, batch_size=32, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_image_datasets, batch_size=32, shuffle=True)


#Imports cat_to_name.json file
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
if gpu != 'gpu':
    print("Please retry with argument --gpu gpu")
    exit()
    
# TODO: Build and train your network
def CreateModel(arch='vgg16', hidden_layer=1024):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        in_size = 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        in_size = 1024
    elif arch == 'alexnet':
        model = models.alexnet(pretrained=True)
        in_size = 9216
    else:
        print("Architecture input not supported.\n" + "Please try again with 'vgg16' or 'densenet121' or 'alexnet'")
        exit()

    dropout=0.5
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([('dropout', nn.Dropout(dropout)),
                                            ('fc1', nn.Linear(in_size, hidden_layer)),
                                            ('relu1', nn.ReLU()),
                                            ('fc2', nn.Linear(hidden_layer, 256)),
                                            ('output', nn.Linear(256, 102)),
                                            ('softmax', nn.LogSoftmax(dim = 1))]))

    model.classifier = classifier

    return model

model = CreateModel(arch, hidden_layer)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr)


def TrainModel(model = model, optimizer = optimizer, criterion = criterion, train_dataloader = train_dataloader, epochs = epochs, gpu = gpu):
    steps_to_print = 10
    steps = 0

    model.to('cuda')
    
    for epoch in range(epochs):
        running_loss = 0
        
        for inputs, labels in train_dataloader:
            steps += 1
            
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if steps % steps_to_print == 0:
                model.eval()
                failed = 0
                accuracy = 0
                
                for valid_input, valid_label in valid_dataloader:
                    valid_input, valid_label = valid_input.to('cuda'), valid_label.to('cuda')
                
                with torch.no_grad():    
                    valid_outputs = model.forward(valid_input)
                    failed = criterion(valid_outputs, valid_label)
                    ps = torch.exp(valid_outputs)
                    top_p, top_class = ps.topk(1, dim = 1)
                    equality = (top_class == valid_label.view(*top_class.shape))
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                    
                print(f"---- Epoch: {epoch+1}/{epochs} ----",
                      "Loss: {:.4f} ----".format(running_loss/steps_to_print),
                      "Validation Lost: {:.4f} ----".format(failed),
                      "Accuracy: {:.4f} ----".format(accuracy))
                
                running_loss = 0
                model.train()
                
TrainModel(model, optimizer, criterion, train_dataloader, epochs, gpu)


# TODO: Do validation on the test set
def CheckAccuracy(model, test_dataloader, gpu):
    model.to('cuda')
    passed = 0
    total = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_input, test_label = test_input.to('cuda'), test_label.to('cuda')
            total += test_label.size(0)
            output = model(test_input)
            predicted = torch.max(output.data, 1)[1]
            passed += (predicted == test_label).sum().item()

    print(f'Accuracy on the test images: {round((passed/total)*100, 2)}')
    
CheckAccuracy(model, test_dataloader, gpu)


# TODO: Save the checkpoint 
model.class_to_idx = train_image_datasets.class_to_idx
torch.save({'arch' : arch,
            'lr': lr,
            'hidden_layer': hidden_layer,
            'gpu': gpu,
            'epochs': epochs,
            'classifier': model.classifier,
            'state_dict' : model.state_dict(),
            'class_to_idx' : model.class_to_idx
           }, 'checkpoint.pth')