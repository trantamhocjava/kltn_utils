import torch
from torchvision.transforms import v2

SEEDING = 42
METRIC_MAX = ("val_c_acc_overall", "val_c_acc", "val_y_acc", "val_y_bmac")
PREPROCESS_LIST = [
    v2.Resize(size=224, interpolation="bicubic", max_size=None, antialias=True),
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]
