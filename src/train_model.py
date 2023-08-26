import torch
from helper import get_formatted_time


def train_model(model, criterion, optimizer, train_loader, valid_loader, epochs, run_on_gpu):
    """
    Train a model on a dataset.

    Args:
        model: The model to train.
        criterion: The loss function.
        optimizer: The optimizer.
        train_loader: The training data loader.
        valid_loader: The validation data loader.
        epochs: The number of epochs to train for.

    Returns:
        The trained model.
    """

    # Move the model to the GPU if available.
    if run_on_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        model.to(device)
        print("GPU is available and being used")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")

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
        print(f" {get_formatted_time()} validation start")
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

    print(f"{get_formatted_time()} End training time:")
    return model
