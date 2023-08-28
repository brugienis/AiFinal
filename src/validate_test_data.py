from datetime import datetime

import helper
import torch


def validate_test_data(model, test_loader, run_on_gpu):
    """
    Test the model.

    :param run_on_gpu: boolean
    :param model:
    :param test_loader: The test data loader.
    """

    # Move the model to the GPU if available.
    if run_on_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        print("GPU is available and being used")
    else:
        device = torch.device("cpu")
        if run_on_gpu:
            print("GPU is not available, using CPU instead")
        else:
            print("using CPU as requested")

    print("Start test time:", helper.get_formatted_time())

    model.to(device)
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test accuracy: {accuracy / len(test_loader):.3f}")

    print("End test time:", helper.get_formatted_time())
