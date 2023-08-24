"""
Train_classifier.
"""

import torch
# from torch import nn

from helper import get_formatted_time


def train_classifier(model, criterion, optimizer, train_loader, valid_loader, test_loader,
                     device_to_use, epochs=1):
    '''
    Train model's classifier.
    '''

    print("\n*** train_classifier: device_to_use: {} ***\n".format(device_to_use))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available and being used")
    else:
        device = torch.device("cpu")
        print("GPU is not available, using CPU instead")

    if device != 'cuda':
        print("\n*** train_classifier: running on CPU not GPU - processing will take a long time ***\n")
    else:
        print("\n*** train_classifier: running on GPU ***\n")

    print("\n*** train_classifier: epochs: {} ***\n".format(epochs))

    model.to(device)

    #     now = datetime.now() # current date and time
    #     time = now.strftime("%H:%M:%S")
    #     print("Start training time:", time)

    print("Start training time:", get_formatted_time())
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            #             print("training processing batch {}". format(steps))
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                #                 with torch.no_grad():
                #                     for inputs, labels in valid_loader:
                #                         inputs, labels = inputs.to(device), labels.to(device)
                #                         logps = model.forward(inputs)
                #                         batch_loss = criterion(logps, labels)

                #                         test_loss += batch_loss.item()

                #                         # Calculate accuracy
                #                         ps = torch.exp(logps)
                #                         top_p, top_class = ps.topk(1, dim=1)
                #                         equals = top_class == labels.view(*top_class.shape)
                #                         accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(get_formatted_time(), " - ", steps, " "
                                                          f"Epoch {epoch + 1}/{epochs}.. "
                                                          f"Train loss: {running_loss / print_every:.3f}.. "
                                                          f"Test loss: {test_loss / len(valid_loader):.3f}.. "
                                                          f"Test accuracy: {accuracy / len(valid_loader):.3f}")
                running_loss = 0
                model.train()

    #     now = datetime.now() # current date and time
    #     time = now.strftime("%H:%M:%S")
    print("End training time:", get_formatted_time())

    return model, epochs
