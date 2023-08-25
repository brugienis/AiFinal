from datetime import datetime

import helper
import torch


def validate_test_data(model, test_loader, device):
    '''
    Test network.
    '''

    print(helper.get_formatted_time(), "validate_test_data start")
    if device != 'cuda':
        print("\n*** validate_test_data: running on CPU not GPU - processing will take a long time ***\n")
    else:
        print("\n*** validate_test_data: running on GPU ***\n")

    now = datetime.now()  # current date and time
    time = now.strftime("%H:%M:%S")
    print("Start test time:", time)

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
    now = datetime.now()  # current date and time
    print(helper.get_formatted_time(), "validate_test_data end")
    time = now.strftime("%H:%M:%S")
    print("End test time:", time)
