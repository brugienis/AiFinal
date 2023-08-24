import argparse


def get_run_type_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, default='checkpoints/',
                        help='path to folder of checkpoints')
    parser.add_argument('--arch', default='vgg', choices=['vgg', 'alexnet', 'resnet'],
                        help='the CNN model architecture')
    parser.add_argument('--learning_rate', default='0.01', choices=['vgg', 'alexnet', 'resnet'],
                        help='learning rate')
    parser.add_argument('--hidden_units', default='3',
                        help='number of hidden units')
    parser.add_argument('--epochs', default='3',
                        help='number of epochs')
    return parser.parse_args()
