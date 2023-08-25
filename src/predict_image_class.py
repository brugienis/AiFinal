import torch

from process_image import process_image


def predict_image_class(image_path, model, topk=5):
    '''
    Predict the class from an image file
    '''

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

    # image_cat = get_image_category(run_type_args.image_path)
    print()

    return top_prob_array, top_classes
