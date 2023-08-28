import os

import torch


def save_checkpoint(architecture, model, classifier_definition, optimizer, class_to_idx, epochs, learning_rate,
                    save_dir, checkpoint_name):
    """
    Save checkpoint file containing model and other objects.

    :param architecture:
    :param model:
    :param classifier_definition:
    :param optimizer:
    :param class_to_idx:
    :param epochs:
    :param learning_rate:
    :param save_dir:
    :param checkpoint_name:
    """
    checkpoint = {
        'architecture': architecture,
        'classifier_definition': classifier_definition,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'training_epochs': epochs,
        'learning_rate': learning_rate
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(checkpoint, os.path.join(save_dir, checkpoint_name))
