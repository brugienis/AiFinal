"""
Save checkpoint.
"""
import os

import torch


# TODO pass folder for checkpoint and save checkpoint in it
def save_checkpoint(architecture, model, classifier_definition, optimizer, class_to_idx, epochs, learning_rate,
                    save_dir, checkpoint_name):
    checkpoint = {
        'architecture': architecture,
        'classifier_definition': classifier_definition,
        'state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'training_epochs': epochs,
        'learning_rate': learning_rate
    }

    # isExist = os.path.exists(save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # model_label = classifier(os.path.join(images_dir, key), model)
    # torch.save(checkpoint, checkpoint_name)
    torch.save(checkpoint, os.path.join(save_dir, checkpoint_name))
