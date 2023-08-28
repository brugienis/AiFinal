import helper
from get_run_type_args import get_run_type_args
from load_saved_checkpoint import load_saved_checkpoint
from load_data import load_data
from predict_image_class import predict_image_class
from print_report import print_report
from validate_test_data import validate_test_data


def predict():
    """
    Get prediction details for the image specified in the argument 'image_path'.
    """
    print(helper.get_formatted_time(), "Predict app start")
    run_type_args, checkpoint_file_name = get_run_type_args('predict')
    run_on_gpu = False if run_type_args.gpu == "No" else True
    train_loader, valid_loader, test_loader, class_to_idx = load_data()
    model = load_saved_checkpoint(run_type_args.save_dir, checkpoint_file_name)
    # validate_test_data(model, test_loader, 'cpu')
    top_prob_array, top_classes = predict_image_class(run_type_args.image_path, model, run_type_args.top_k, run_on_gpu)
    print_report(run_type_args.image_path, top_prob_array, top_classes)
    print(helper.get_formatted_time(), "Predict app end")


if __name__ == "__main__":
    predict()
