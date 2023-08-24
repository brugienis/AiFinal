import helper
from get_run_type_args import get_run_type_args
# from src.predict import main


def predict():
    print(helper.get_formatted_time(), "Predict app start")
    run_type_args = get_run_type_args('predict')
    print(run_type_args)
    run_type_args = get_run_type_args('predict')
    if run_type_args.image_path == None:
        print("Image path missing")
    print(helper.get_formatted_time(), "Predict app end")


print("hola", __name__)
# Call to main function to run the program
# if __name__ == "__main__":
if __name__ == "__main__":
    predict()