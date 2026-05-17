import random

import numpy as np
import torch


def random_select(
    concept2cls,
    num_concepts,
    num_images_per_class,
):
    num_cls = len(num_images_per_class)
    selected_idx = []
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    concept2cls = torch.from_numpy(concept2cls).long()

    for i in range(num_cls):
        cls_idx = torch.where(concept2cls == i)[0]

        if len(cls_idx) == 0:
            continue

        elif len(cls_idx) < num_concepts_per_cls:
            global_idx = cls_idx
        else:
            global_idx = random.sample(cls_idx.tolist(), num_concepts_per_cls)

        selected_idx.extend(global_idx)

    return torch.tensor(selected_idx)
