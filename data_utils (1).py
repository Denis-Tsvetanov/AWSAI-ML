import torch
from torchvision import datasets, transforms

def load_and_process_data(data_dir):
    print(torch.cuda.is_available())
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    image_datasets = {x: datasets.ImageFolder(data_dir + '/' + x, transform=data_transforms[x]) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True) for x in ['train', 'test']}
    class_to_idx = image_datasets['train'].class_to_idx
    
    return dataloaders, class_to_idx