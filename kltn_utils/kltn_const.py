import torch
from torchvision.transforms import InterpolationMode, v2

SEEDING = 42
METRIC_MAX = ("val_c_acc_overall", "val_c_acc", "val_y_acc", "val_y_bmac")
PREPROCESS_LIST = [
    v2.Resize(
        size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True
    ),
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]
CLIP_MODEL_FROM_OPENAI = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"]
CLIP_MODEL_FROM_HF_HUB = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
)
