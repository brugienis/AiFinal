import helper
from torch import nn
from torch import optim
from torchvision import models
from get_classifier_definition import get_classifier_definition


def define_model(model_architecture, hidden_units, learning_rate):
    """
    Define model.

    :param model_architecture: pre-trained model name
    :param hidden_units: int
    :param learning_rate: float
    :return: model with classifier layer redefined
    """

    features_layer_size = 0
    model = None
    if model_architecture == "densenet":
        model = models.densenet121(pretrained=True)
        features_layer_size = 1024
    elif model_architecture == "resnet":
        model = models.resnet101(pretrained=True)
        features_layer_size = 2048

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier_definition = get_classifier_definition(features_layer_size, hidden_units)
    if model_architecture == "resnet":
        model.fc = nn.Sequential(classifier_definition)
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)
    else:
        model.classifier = nn.Sequential(classifier_definition)
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    return model, criterion, optimizer, classifier_definition
