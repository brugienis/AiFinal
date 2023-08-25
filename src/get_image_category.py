import json


def get_image_category(image_path):
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    image_cat = image_path.split("/")[2]
    print("image_cat:", image_cat)
    return cat_to_name[image_cat]