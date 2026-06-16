import json

import torch

from .. import kltn_utils


def load_json(package_file):
    with package_file.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_label_concept(dataset_dir, file_names, class_names, concept2class, transform):
    file_paths = [f"{dataset_dir}/{item}" for item in file_names]
    imgs = []
    for file_path in file_paths:
        img = kltn_utils.read_img(file_path)
        img = transform(img)
        imgs.append(img)

    img = torch.stack(imgs, dim=0)

    label = dataset_dir.split("/")[-1]
    label_index = class_names.index(label)

    class2concept = kltn_utils.build_class_concept_matrix(
        concept2class, len(class_names)
    )
    concept = class2concept[label_index]

    return img, label, concept, file_paths
