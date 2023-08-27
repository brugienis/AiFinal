"""
Define model.
"""
from collections import OrderedDict
import helper
from torch import nn
from torch import optim
from torchvision import models

from build_hidden_layers import build_hidden_layers


def define_model(model_architecture, hidden_units, learning_rate):
    print(helper.get_formatted_time(), "define_model start", "model_architecture:", model_architecture,
          "learning_rate:",
          learning_rate)

    c_new = OrderedDict([
        ('fc1', nn.Linear(1024, 256)),  # Input layer
        ('relu1', nn.ReLU()),  # Activation function
        ('drop1', nn.Dropout(0.2))
    ])
    for i in range(hidden_units):
        c_new['hidden' + str(i + 1)] = nn.Linear(256, 256)
        c_new['hrelu' + str(i + 1)] = nn.ReLU()
        if i != hidden_units - 1: c_new['hdrop' + str(i + 1)] = nn.Dropout(0.2)
    c_new['fc2'] = nn.Linear(256, 1000)
    c_new['output'] = nn.LogSoftmax(dim=1)
    print("c_new:", c_new)

    # classifier_definition = OrderedDict([
    #     ('fc1', nn.Linear(1024, 256)),
    #     ('relu', nn.ReLU()),
    #     ('drop', nn.Dropout(0.2)),
    #     ('fc2', nn.Linear(256, 1000)),
    #     ('output', nn.LogSoftmax(dim=1))
    # ])
    # Define the new classifier structure
    # todo 'fc2' change 100 to 102
    classifier_definition = OrderedDict([
        ('fc1', nn.Linear(1024, 256)),  # Input layer
        ('relu1', nn.ReLU()),  # Activation function
        ('drop1', nn.Dropout(0.2)),
        ('hidden1', nn.Linear(256, 256)),  # First hidden layer
        ('hrelu1', nn.ReLU()),  # Activation function
        ('hdrop1', nn.Dropout(0.2)),
        ('hidden2', nn.Linear(256, 256)),  # Second hidden layer
        ('hrelu2', nn.ReLU()),  # Activation function
        ('fc2', nn.Linear(256, 1000)),  # Output layer
        ('output', nn.LogSoftmax(dim=1))
    ])
    classifier_definition_2048 = OrderedDict([
        ('fc1', nn.Linear(2048, 256)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.2)),
        ('fc2', nn.Linear(256, 1000)),
        ('output', nn.LogSoftmax(dim=1))
    ])

    if model_architecture == "densenet":
        model = models.densenet121(pretrained=True)
    elif model_architecture == "resnet":
        model = models.resnet101(pretrained=True)
    else:
        # TODO: Stop processing at that point
        print("define_model - NO CODE TO HANDLE model_architecture:", model_architecture)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    if model_architecture == "resnet":
        model.fc = nn.Sequential(classifier_definition_2048)
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        # model.classifier = nn.Sequential(classifier_definition)
        model.classifier = nn.Sequential(c_new)
        # Only train the classifier parameters, feature parameters are frozen
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    # print("final model", model)
    criterion = nn.NLLLoss()

    print(helper.get_formatted_time(), "define_model end")
    return model, criterion, optimizer, \
        classifier_definition_2048 if model_architecture == "resnet" else classifier_definition
