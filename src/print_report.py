import json


def print_report(image_path, top_prob_array, top_classes):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    image_cat = image_path.split("/")[2]
    image_label = cat_to_name[image_cat]
    print("-"*70)
    print("Image label:", image_label, " | image_path:", image_path)
    print("{:30}: {}".format("               Class", "Probability\n"))
    for i in range(len(top_prob_array)):
        print("{:>30}:       {:5.2f}".format(cat_to_name[top_classes[i]], top_prob_array[i] * 100))
    print("-"*70)


# print("Real: {:>26}   Classifier: {:>30}".format(results_dic[key][0],
