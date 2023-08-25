import helper
from get_run_type_args import get_run_type_args
from load_saved_checkpoint import load_saved_checkpoint
from load_data import load_data
from predict_image_class import predict_image_class
from get_image_category import get_image_category
from validate_test_data import validate_test_data


# from src.predict import main


def predict():
    print(helper.get_formatted_time(), "Predict app start")
    run_type_args = get_run_type_args('predict')
    print(run_type_args)
    run_type_args = get_run_type_args('predict')
    if run_type_args.image_path == None:
        print("Image path missing")
    train_loader, valid_loader, test_loader, class_to_idx = load_data()
    model = load_saved_checkpoint('final_project_checkpoint.pth')
    # validate_test_data(model, test_loader, 'cpu')
    top_prob_array, top_classes = predict_image_class(run_type_args.image_path, model, topk=run_type_args.top_k)
    image_cat = get_image_category(run_type_args.image_path)
    print("image_cat:", image_cat, "top_classes:", top_classes)
    print(helper.get_formatted_time(), "Predict app end")


print("hola", __name__)
# Call to main function to run the program
# if __name__ == "__main__":
if __name__ == "__main__":
    predict()
