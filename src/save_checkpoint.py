"""
Save checkpoint.
"""
import torch


def save_checkpoint(architecture, model, classifier_definition, optimizer, class_to_idx, epochs, checkpoint_name):
    checkpoint = {
        'architecture': architecture,
        'classifier_definition': classifier_definition,
        # 'classifier': model.classifier,
        'state_dict': model.state_dict(),
        # 'class_to_idx': model.class_to_idx,
        'class_to_idx': class_to_idx,
        'optimizer_state': optimizer.state_dict(),
        'training_epochs': epochs
    }
    torch.save(checkpoint, checkpoint_name)
