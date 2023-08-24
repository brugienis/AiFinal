"""
Load data to train, validate and test model.
"""

import torch
from torchvision import datasets, transforms, models

from src import helper


def load_data():
    print(helper.get_formatted_time(), "load_data start")
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
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

    # TODO: Load the datasets with ImageFolder
    # image_datasets =
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    # dataloaders =
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
    print("load done")
    print(helper.get_formatted_time(), "load_data end")
    return train_loader, valid_loader, test_loader, train_data.class_to_idx
