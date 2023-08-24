import argparse


def get_run_type_args(args_for_class):
    parser = argparse.ArgumentParser()

    if args_for_class == 'train':
        parser.add_argument('--save_dir', type=str, default='checkpoints',
                            help='path to folder of checkpoints')
        parser.add_argument('--arch', default='vgg', choices=['vgg', 'alexnet', 'resnet'],
                            help='the CNN model architecture')
        parser.add_argument('--learning_rate', type=float, default='0.001', choices=['vgg', 'alexnet', 'resnet'],
                            help='learning rate')
        parser.add_argument('--hidden_units', default='3',
                            help='number of hidden units')
        parser.add_argument('--epochs', type=int, default='2',
                            help='number of epochs')
    else:
        parser.add_argument('--image_path', type=str,
                            help='path to image')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                            help='path to checkpoint folder')
        parser.add_argument('--top_k', default='3',
                            help='return top K classes')

    return parser.parse_args()
