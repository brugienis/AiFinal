import torch
from helper import get_formatted_time
import helper


def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs, run_on_gpu):
    """
    Train a model on a dataset.

    :param model: The model to train.
    :param criterion: The loss function.
    :param optimizer: The optimizer.
    :param train_loader: The training data loader.
    :param valid_loader: The validation data loader.
    :param epochs: The number of epochs to train for.
    :param run_on_gpu: If true, process function on GPU
    :return: The trained model.
    """

    print(helper.get_formatted_time(), "train_model start. run_on_gpu:", run_on_gpu)
    # Move the model to the GPU if available.
    if run_on_gpu and torch.cuda.is_available():
        print(helper.get_formatted_time(), "train_model before: device = torch.device('cuda')")
        device = torch.device("cuda")
        print(helper.get_formatted_time(), "train_model before: model.to(device)")
        model.to(device)
        print(helper.get_formatted_time(), "GPU is available and being used")
    else:
        device = torch.device("cpu")
        if run_on_gpu:
            print("GPU is not available, using CPU instead")
        else:
            print("using CPU as requested")

    print("\n*** train_model: epochs: {} ***\n".format(epochs))

    # Start training
    print("Start training time:", get_formatted_time())
    running_loss = 0
    for epoch in range(epochs):
        print(f"{get_formatted_time()} epoch {epoch + 1}/{epochs} start")
        model.train()
        for inputs, labels in train_loader:
            # Move the input and label tensors to the GPU.
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            # print("before loss.backward()")
            loss.backward()
            # print("after  loss.backward()")
            optimizer.step()
            running_loss += loss.item()

        # Evaluate the model on the validation set
        print(f"{get_formatted_time()} validation start")
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_accuracy = 0
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()

                # Calculate accuracy
                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        print(f"{get_formatted_time()} Epoch {epoch + 1}/{epochs}.. "
              f"Train loss: {running_loss / len(train_loader):.3f}.. "
              f"Validation loss: {val_loss / len(valid_loader):.3f}")
        print(f"Validation accuracy: {val_accuracy / len(valid_loader):.3f}")

    device = torch.device("cpu")
    model.to(device)
    print(f"{get_formatted_time()} End training time:")
    print(helper.get_formatted_time(), "train_model end")
    return model
