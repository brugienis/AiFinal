"""
Load model from the checkpoint file.
"""
import helper
import torch
from torch import nn
from torch import optim
from torchvision import models


def load_saved_checkpoint(checkpoint_name):
    print(helper.get_formatted_time(), "load_saved_checkpoint start", "checkpoint name", checkpoint_name)

    checkpoint = torch.load('final_project_checkpoint.pth')
    architecture = checkpoint['architecture']
    print("load_saved_checkpoint architecture: ", architecture)
    #     print("classifier_definition:", checkpoint['classifier_definition'])

    # if checkpoint['architecture'] == 'densenet121':
    classifier_definition = checkpoint['classifier_definition']
    print("classifier_definition:", classifier_definition)
    model = models.densenet121(pretrained=True) if architecture == "densenet" else models.resnet101(pretrained=True)

    if architecture == "resnet":
        model.fc = nn.Sequential(classifier_definition)
        optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    else:
        model.classifier = nn.Sequential(classifier_definition)
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    print(model)

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    # ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^
    # File
    # "/Users/bohdan/PycharmProjects/udacityAiFinal/src/load_saved_checkpoint.py", line
    # 30, in load_saved_checkpoint
    # model.load_state_dict(checkpoint['state_dict'])
    # File
    # "/Users/bohdan/PycharmProjects/udacityAiFinal/lib/python3.11/site-packages/torch/nn/modules/module.py", line
    # 2041, in load_state_dict
    # raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
    #     RuntimeError: Error(s) in loading
    # state_dict
    # for DenseNet:
    #     Unexpected
    # key(s) in state_dict: "classifier.hidden1.weight", "classifier.hidden1.bias", "classifier.hidden2.weight", "classifier.hidden2.bias".
    # size
    # mismatch
    # for classifier.fc1.weight: copying
    # a
    # param
    # with shape torch.Size([256, 1024]) from checkpoint, the shape in current model is torch.Size([256, 2048]).
    #
    # Process finished w

    # optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    epochs = checkpoint['training_epochs']
    #     steps = checkpoint['training_steps']
    #     running_loss = checkpoint['training_loss']
    # model.eval()

    # return other variables if required in the future
    print(helper.get_formatted_time(), "load_saved_checkpoint end")
    return model
