"""
Train model.
"""
from get_run_type_args import get_run_type_args


def main():
    print("hello start")
    run_type_args = get_run_type_args()
    print("save_dir:", run_type_args.save_dir)
    print(run_type_args)


# Call to main function to run the program
if __name__ == "__main__":
    main()
