"""
Define and train model.
"""
from get_run_type_args import get_run_type_args
import helper
from define_model import define_model
from load_data import load_data
from src.save_checkpoint import save_checkpoint
from train_classifier import train_classifier
from validate_test_data import validate_test_data


def main():
    print(helper.get_formatted_time(), "App start")
    run_type_args = get_run_type_args()
    print("save_dir:", run_type_args.save_dir)
    print(run_type_args)
    train_loader, valid_loader, test_loader, class_to_idx = load_data()
    model, criterion, optimizer, classifier_definition = define_model()
    model, epochs = train_classifier(model, criterion, optimizer, train_loader, valid_loader, test_loader,
                                     device_to_use='cpu', epochs=1)
    validate_test_data(model, test_loader, 'cpu')

    model.class_to_idx = class_to_idx
    save_checkpoint(model, classifier_definition, optimizer, class_to_idx, epochs, 'final_project_checkpoint.pth')
    print(helper.get_formatted_time(), "App end")


# Call to main function to run the program
if __name__ == "__main__":
    main()
