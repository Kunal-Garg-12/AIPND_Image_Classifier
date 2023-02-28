# Imports here
import torch
from torchvision import transforms, models
import torchvision.models as models
from PIL import Image
import json
import argparse

#Take arguments from CLI
parser = argparse.ArgumentParser()

parser.add_argument('test_file', type=str, default='flowers/test/69/image_05959.jpg')
parser.add_argument('--checkpoint_file', type=str, default='checkpoint.pth')
parser.add_argument('--category_names', type=str, default='cat_to_name.json')
parser.add_argument('--top_k', type=int, default=5)
parser.add_argument('--gpu', default='gpu', type=str)

inputs = parser.parse_args()

test_file = inputs.test_file
category_names = inputs.category_names
checkpoint_file = inputs.checkpoint_file
topk = inputs.top_k
gpu = inputs.gpu

#Imports inputted JSON file
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)


# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(path):
    checkpoint = torch.load(path)
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
        
    return model
    
model = load_model(checkpoint_file)


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    adjust_img = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])
                                    ])
    return adjust_img(img).float()


def predict(image_path=test_file, model=model, topk=topk, gpu=gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    if gpu == 'gpu':
        model.to('cuda')
    img = process_image(image_path)
    img = img.float().unsqueeze_(0)

    with torch.no_grad():
        if gpu == 'gpu':
            img = img.to('cuda')
        else:
            print("Please retry with argument --gpu gpu")
            exit()
        output = model.forward(img)

    probability = torch.nn.functional.softmax(output.data, dim = 1)

    probs, indices = probability.topk(topk)
    probs = probs.cpu().numpy()[0]
    indices = indices.cpu().numpy()[0]

    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]

    return probs, classes

results = predict(test_file, model, topk, gpu)
names = [cat_to_name[str(x)] for x in results[1]]
print("FLOWER NAME".center(30, "="), ":", "PROBABILITY".center(30, "="))
for i in range(topk):
    print(names[i].ljust(30), ":", f"{round(results[0][i] * 100, 2)} %".rjust(30))