"""
Define and train model.
"""
from get_run_type_args import get_run_type_args
from src import helper
from src.define_model import define_model
from src.load_data import load_data
from src.train_classifier import train_classifier


def main():
    print(helper.get_formatted_time(), "hello start")
    run_type_args = get_run_type_args()
    print("save_dir:", run_type_args.save_dir)
    print(run_type_args)
    train_loader, valid_loader, test_loader = load_data()
    model, criterion, optimizer = define_model()
    # model = define_model(train_loader, valid_loader, test_loader)
    model = train_classifier(model, criterion, optimizer, train_loader, valid_loader, test_loader,
                     device_to_use='cpu', epochs=1)


# Call to main function to run the program
if __name__ == "__main__":
    main()
