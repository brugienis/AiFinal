"""
Define model. DELETE IT - not needed
"""
from collections import OrderedDict

import helper
from torch import nn
from torch import optim
from torchvision import models


def define_model(model_architecture, learning_rate):
    print(helper.get_formatted_time(), "define_model start", "model_architecture:", model_architecture, "learning_rate:",
          learning_rate)

    classifier_definition = OrderedDict([
        ('fc1', nn.Linear(1024, 256)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.2)),
        ('fc2', nn.Linear(256, 1000)),
        ('output', nn.LogSoftmax(dim=1))
    ])
    classifier_definition_2048 = OrderedDict([
        ('fc1', nn.Linear(2048, 1000)),
        ('relu', nn.ReLU()),
        ('drop', nn.Dropout(0.2)),
        ('fc2', nn.Linear(1000, 1000)),
        ('output', nn.LogSoftmax(dim=1))
    ])
    criterion = nn.NLLLoss()

    model = None
    optimizer = None
    if model_architecture == "densenet":
        #  classifier): Linear(in_features=1024, out_features=1000, bias=True)
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Sequential(classifier_definition)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    elif model_architecture == "vgg":
        model = models.vgg16(pretrained=True)
        model.fc = nn.Sequential(classifier_definition)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    # elif model_architecture == "alexnet":
    #     model = models.densenet121(pretrained=True)
    elif model_architecture == "resnet":
        model = models.resnet101(pretrained=True)
        print("model resnet before replacing the classifier:", model)
        model.fc = nn.Sequential(classifier_definition_2048)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        print("define_model - NO CODE TO HANDLE model_architecture:", model_architecture)
    # model = models.resnet50(pretrained=True)
    # model = models.densenet121(pretrained=True)
    # model = models.resnet101(pretrained=True)
    # print("model before classifier replaced\n", model)

    print("model after replacing the classifier:", model)
    # Freeze parameters so we don't backprop through them

    for param in model.parameters():
        param.requires_grad = False

    # model.classifier = nn.Sequential(classifier_definition)
    # print(model)
    # criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    # optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

    # optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    print(helper.get_formatted_time(), "define_model end")
    return model, criterion, optimizer, classifier_definition
