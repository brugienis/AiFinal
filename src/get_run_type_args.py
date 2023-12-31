import argparse


def get_run_type_args(args_for_class):
    """

    :param args_for_class:
    :return:
    """
    parser = argparse.ArgumentParser()

    if args_for_class == 'train':
        parser.add_argument('--arch', default='densenet', choices=['densenet', 'resnet'],
                            help='the CNN model architecture')
        parser.add_argument('--learning_rate', type=float, default='0.001',
                            help='learning rate')
        parser.add_argument('--hidden_units', type=int, default='2',
                            help='number of hidden units')
        parser.add_argument('--epochs', type=int, default='3',
                            help='number of epochs')
        # parser.add_argument('--gpu', default='No', action='store_true', help='utilize GPU')
    else:
        parser.add_argument('--image_path', type=str, default='flowers/test/2/image_05133.jpg',
                            help='path to image')
        parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                            help='path to checkpoint folder')
        parser.add_argument('----category_names', type=str, default='cat_to_name.json',
                            help='category to name JSON file')
        parser.add_argument('--top_k', type=int, default='3',
                            help='return top K classes')

    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='path to folder of checkpoints')
    parser.add_argument('--gpu', default='Yes', action='store_true', help='utilize GPU')
    args = parser.parse_args()
    print("-" * 50)
    print("                Run time params")
    for arg in vars(args):
        arg_value = getattr(args, arg)
        if arg == "gpu":
            arg_value = False if arg_value == "No" else True
        print("{:30}: {}".format(arg, arg_value))
    print("-" * 50)
    checkpoint_file_name = "final_project_checkpoint.pth"
    return args, checkpoint_file_name
