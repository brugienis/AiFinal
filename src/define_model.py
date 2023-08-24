"""
Define model.
"""
from collections import OrderedDict

from torch import nn
from torch import optim
from src import helper
from torchvision import models


def define_model():
    print(helper.get_formatted_time(), "define_model start")
    # model = models.resnet50(pretrained=True)
    model = models.densenet121(pretrained=True)
    # model = models.resnet101(pretrained=True)
    # model

    # Freeze parameters so we don't backprop through them

    for param in model.parameters():
        param.requires_grad = False

    classifier_definition = OrderedDict([
        ('fc1', nn.Linear(1024, 256)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.2)),
        ('fc2', nn.Linear(256, 1000)),
        ('output', nn.LogSoftmax(dim=1))
    ])

    model.classifier = nn.Sequential(classifier_definition)
    # print(model)
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    # optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    print(helper.get_formatted_time(), "define_model end")
    return model, criterion, optimizer