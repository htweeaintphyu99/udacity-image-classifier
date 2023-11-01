import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt


data_directory = 'flowers'
train_dir = data_directory + '/train'
valid_dir = data_directory + '/valid'
test_dir = data_directory + '/test'

# TODO: Define your transforms for the training, validation, and testing sets
train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# TODO: Load the datasets with ImageFolder
train_datasets = ImageFolder(train_dir, transform = train_transforms)
valid_datasets = ImageFolder(valid_dir, transform = valid_transforms)
test_datasets = ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = DataLoader(train_datasets, batch_size=32, shuffle=True)
validloader = DataLoader(valid_datasets, batch_size=32)
testloader = DataLoader(test_datasets, batch_size=32)


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

    
#Set up your network 
def model_setup(arch, dropout, hidden_units, lr, gpu):
    # TODO: Build and train your network
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True) 
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = Network(input_size = 25088,  
                                 output_size = 102,   
                                 hidden_layers = [hidden_units], dropout = dropout)  

    elif arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = Network(input_size = 1024,  
                                 output_size = 102,   
                                 hidden_layers = [hidden_units], dropout = dropout)  
    elif arch == 'alexnet':
        model = models.alexnet(pretrained = True)
        for param in model.parameters():
            param.requires_grad = False
        
        classifier = Network(input_size = 9216,  
                                 output_size = 102,   
                                 hidden_layers = [hidden_units], dropout = dropout)  
    else:
        print("Your chosen architecture is not supported here. Choose one of vgg16, densenet121 or alexnet!")
        
    

    
    model.classifier = classifier

    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)

    device = torch.device('cuda' if torch.cuda.is_available() and gpu == 'gpu' else 'cpu')
    device = torch.device(device)
    
    torch.cuda.empty_cache()
    model.to(device);
    
    return model, optimizer


#Train your network
def train(model, epochs, optimizer, gpu): 
    steps = 0
    running_loss = 0
    print_every = len(trainloader) // 2
    criterion = nn.NLLLoss()
    device = torch.device('cuda' if torch.cuda.is_available() and gpu == 'gpu' else 'cpu')
    device = torch.device(device)
    
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    return model


# TODO: Do validation on the test set
def validate_testset(model, gpu):
    criterion = nn.NLLLoss()
    test_loss = 0
    test_accuracy = 0
    model.eval() 
    device = torch.device('cuda' if torch.cuda.is_available() and gpu == 'gpu' else 'cpu')
    device = torch.device(device)
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}.. "
          f"Test accuracy: {test_accuracy/len(testloader):.3f}")
  

# TODO: Save the checkpoint 
def save_checkpoint(model, arch, dropout, epochs, save_dir, hidden_units, optimizer):
    checkpoint = {
        'arch': arch, 
        'dropout': dropout,
        'output_size': 102,   
        'hidden_units': hidden_units,
        'state_dict': model.state_dict(),  
        'class_to_idx': train_datasets.class_to_idx,  
        'epochs': epochs,  
        'optimizer_state_dict': optimizer.state_dict()  
    }

    torch.save(checkpoint, save_dir)
    
    
# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(save_dir):
    checkpoint = torch.load(save_dir)
    output_size = 102  

    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained = True)  
        classifier = Network(input_size = checkpoint[25088],  
                             output_size = 102,  
                             hidden_layers = [checkpoint['hidden_units']], dropout = checkpoint['dropout']) 
    
    elif checkpoint['arch'] == 'densenet121':
        model = models.densenet121(pretrained = True)
        classifier = Network(input_size = checkpoint[1024],  
                             output_size = 102,  
                             hidden_layers = [checkpoint['hidden_units']], dropout = checkpoint['dropout']) 
    
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained = True)
        classifier = Network(input_size = checkpoint[9216],  
                             output_size = 102,  
                             hidden_layers = [checkpoint['hidden_units']], dropout = checkpoint['dropout']) 
    
        
    classifier_layers = []
        
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


# TODO: Process a PIL image for use in a PyTorch model
def process_image(image_path):
    
    image = Image.open(image_path)
    
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    processed_image = image_transforms(image)
    processed_image = processed_image.unsqueeze(0)
    return processed_image


# TODO: Implement the code to predict the class from an image file
def predict(processed_img, model, topk, cat_to_name, gpu):   

    device = torch.device('cuda' if torch.cuda.is_available() and gpu == 'gpu' else 'cpu')
    device = torch.device(device)

    model.to(device)
    img_torch = processed_img.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.to(device))
        
    probability = F.softmax(output.data,dim=1)
    
    top_prob, top_indices = probability.topk(topk)
    
    # Convert indices to class labels
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_class = [idx_to_class[idx.item()] for idx in top_indices[0]]

    # Convert tensors to lists
    top_probs = top_prob[0].cpu().numpy().tolist()
    class_names = [cat_to_name[class_idx] for class_idx in top_class]

    return top_probs, class_names


