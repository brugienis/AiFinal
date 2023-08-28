import torch

from process_image import process_image
import helper


def predict_image_class(image_path, model, topk, run_on_gpu):
    """
    Predict the class from an image file.

    :param image_path:
    :param model:
    :param topk:
    :param run_on_gpu:
    :return: top K probabilities and classes
    """

    print(helper.get_formatted_time(), "predict_image_class start. run_on_gpu:", run_on_gpu)
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

    print("Start predict_image_class time:", helper.get_formatted_time())
    model.eval()
    img_tensor = process_image(image_path)
    image = img_tensor.unsqueeze(0)
    logps = model.forward(image)

    ps = torch.exp(logps)
    top_p, top_classes_idx = ps.topk(topk, dim=1)
    top_prob_array = top_p.data.numpy()[0]
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    top_classes_data = top_classes_idx.data.numpy()
    top_classes_list = top_classes_data[0].tolist()

    top_classes = [idx_to_class[x] for x in top_classes_list]

    print("End predict_image_class time:", helper.get_formatted_time())
    return top_prob_array, top_classes
