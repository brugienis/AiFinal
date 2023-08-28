from get_run_type_args import get_run_type_args
import helper
from define_model import define_model
from load_data import load_data
from save_checkpoint import save_checkpoint
from train_model import train_model
from validate_test_data import validate_test_data


#       python train.py --epochs 1 --arch densenet
#       python predict.py

def main():
    """
    Define and train model.
    """
    print(helper.get_formatted_time(), "Train app start")
    run_type_args, checkpoint_file_name = get_run_type_args('train')
    run_on_gpu = False if run_type_args.gpu == "No" else True
    print("run_type_args.gpu:", run_type_args.gpu, type(run_type_args.gpu), "; run_on_gpu:", run_on_gpu)

    train_loader, valid_loader, test_loader, class_to_idx = load_data()
    model, criterion, optimizer, classifier_definition = define_model(run_type_args.arch, run_type_args.hidden_units,
                                                                      run_type_args.learning_rate)
    model = train_model(model, criterion, optimizer, train_loader, valid_loader, run_type_args.epochs,
                        run_on_gpu)
    validate_test_data(model, test_loader, run_on_gpu)
    model.class_to_idx = class_to_idx
    save_checkpoint(run_type_args.arch, model, classifier_definition, optimizer, class_to_idx, run_type_args.epochs,
                    # run_type_args.learning_rate, run_type_args.save_dir, 'final_project_checkpoint.pth')
                    run_type_args.learning_rate, run_type_args.save_dir, checkpoint_file_name)
    print(helper.get_formatted_time(), "Train app end")


# Call to main function to run the program
if __name__ == "__main__":
    main()
