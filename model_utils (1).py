import torch
from torch import nn
from torchvision import models
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def build_and_train_model(arch, hidden_units, learning_rate, use_gpu, dataloaders, epochs):
    model = build_model(arch, hidden_units)
 
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        model.train()
        for inputs, labels in dataloaders['train']:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                test_loss, accuracy = validate_model(model, criterion, dataloaders['test'], device)
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss:.3f}.. "
                      f"Test accuracy: {accuracy:.3f}")
                running_loss = 0
                model.train()

    return model, optimizer, criterion

def build_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError("Unsupported model architecture")
    
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier
    return model

def validate_model(model, criterion, dataloader, device):
    test_loss = 0
    accuracy = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
    return test_loss / len(dataloader), accuracy / len(dataloader)

def save_checkpoint(model, save_dir, arch, class_to_idx):
    checkpoint = {
        'architecture': arch,
        'class_to_idx': class_to_idx,
        'classifier': model.classifier,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_dir + '/checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if checkpoint['architecture'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.avgpool = nn.AdaptiveAvgPool2d((49, 512))
    else:
        raise ValueError("Unsupported model architecture in the checkpoint")
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model,model.class_to_idx

def predict(image_path, model, topk, category_names, use_gpu):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    model.to(device)
    image = process_image(image_path)
    
    imshow(image)
    model.eval()
    
    image_tensor = torch.from_numpy(image).float()
    image_tensor.to(device)
        
    with torch.no_grad():
        output = model(image_tensor)
        
    ps = torch.exp(output)
    top_p, top_class = ps.topk(topk, dim=1)
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    classes = [cat_to_name[idx_to_class[i.item()]] for i in top_class[0]]
    top_p = top_p[0].tolist()

    top_p = np.array(top_p)
    classes = np.array(classes)

    plt.figure(figsize=(8, 6))
    plt.barh(classes, top_p, color='blue')
    plt.xlabel('Probabilities')
    plt.ylabel('Classes')
    plt.gca().invert_yaxis()
    plt.show()

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((256,256))
    image = image.crop(((256-224)/2,(256-224)/2,(256+224)/2,(256+224)/2))
    np_image = (np.array(image))/255.0
    np_image = (np_image - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    np_image = np_image.transpose(2,0,1)
    
    return np_image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
  
    image = image.transpose((1, 2, 0))
    image = [0.229, 0.224, 0.225] * image + [0.485, 0.456, 0.406]
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
