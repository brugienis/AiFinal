"""
Load model from the checkpoint file.
"""
import os

import helper
import torch
from torch import nn
from torch import optim
from torchvision import models


def load_saved_checkpoint(save_dir, checkpoint_name):
    print(helper.get_formatted_time(), "load_saved_checkpoint start", "checkpoint name", checkpoint_name)

    # checkpoint = torch.load('final_project_checkpoint.pth')
    checkpoint = torch.load(os.path.join(save_dir, checkpoint_name))

    architecture = checkpoint['architecture']
    print("load_saved_checkpoint architecture: ", architecture)

    classifier_definition = checkpoint['classifier_definition']
    model = models.densenet121(pretrained=True) if architecture == "densenet" else models.resnet101(pretrained=True)

    learning_rate = checkpoint['learning_rate']
    if architecture == "resnet":
        model.fc = nn.Sequential(classifier_definition)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        model.classifier = nn.Sequential(classifier_definition)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    optimizer.load_state_dict(checkpoint['optimizer_state'])

    epochs = checkpoint['training_epochs']

    # return other variables if required in the future
    print(helper.get_formatted_time(), "load_saved_checkpoint end")
    return model
