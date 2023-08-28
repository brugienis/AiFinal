from collections import OrderedDict
from torch import nn


def get_classifier_definition(features_layer_size, hidden_units):
    """
    Build the classifier_definition - the last layer of the predefined model.

    :param features_layer_size: int
    :param hidden_units: int
    :rtype: OrderedDict
    """
    # todo 'fc2' change 1000 to 102
    classifier_definition = OrderedDict([
        ('fc1', nn.Linear(features_layer_size, 256)),   # Input layer
        ('relu1', nn.ReLU()),                           # Activation function
        ('drop1', nn.Dropout(0.2))
    ])
    for i in range(hidden_units):
        classifier_definition['hidden' + str(i + 1)] = nn.Linear(256, 256)
        classifier_definition['hrelu' + str(i + 1)] = nn.ReLU()
        if i != hidden_units - 1: classifier_definition['hdrop' + str(i + 1)] = nn.Dropout(0.2)
    classifier_definition['fc2'] = nn.Linear(256, 1000)
    classifier_definition['output'] = nn.LogSoftmax(dim=1)
    print("classifier_definition:", classifier_definition)

    return classifier_definition
