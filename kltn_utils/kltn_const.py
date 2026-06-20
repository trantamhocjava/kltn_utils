import torch
from torchvision.transforms import InterpolationMode, v2

from .clip_model import build_clip_model, get_feat

SEEDING = 42

METRIC_MAX = (
    "val_c_overall_acc",
    "val_c_acc",
    "val_y_acc",
    "val_y_bmac",
)

IMAGE_PREPROCESS_LIST = [
    v2.Resize(
        size=224, interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=True
    ),
    v2.CenterCrop(size=(224, 224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]


CLIP_MODELS = {
    "RN50": {
        "source": "openai",
        "embedding_dim": 1024,
        "visual_feature_dim": 2048,
        "num_heads": 16,
    },
    "RN101": {
        "source": "openai",
        "embedding_dim": 512,
        "visual_feature_dim": 2048,
        "num_heads": 16,
    },
    "RN50x4": {
        "source": "openai",
        "embedding_dim": 640,
        "visual_feature_dim": 2560,
        "num_heads": 16,
    },
    "RN50x16": {
        "source": "openai",
        "embedding_dim": 768,
        "visual_feature_dim": 3072,
        "num_heads": 12,
    },
    "RN50x64": {
        "source": "openai",
        "embedding_dim": 1024,
        "visual_feature_dim": 4096,
        "num_heads": 16,
    },
    "ViT-B-32": {
        "source": "openai",
        "embedding_dim": 512,
        "visual_feature_dim": 768,
        "num_heads": 12,
    },
    "ViT-B-16": {
        "source": "openai",
        "embedding_dim": 512,
        "visual_feature_dim": 768,
        "num_heads": 12,
    },
    "ViT-L-14": {
        "source": "openai",
        "embedding_dim": 768,
        "visual_feature_dim": 1024,
        "num_heads": 16,
    },
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": {
        "source": "hf-hub",
        "embedding_dim": 512,
        "visual_feature_dim": 768,
        "num_heads": 12,
        "logit_scale": 85.2323,
    },
    "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K": {
        "source": "hf-hub",
        "embedding_dim": 768,
        "visual_feature_dim": 1024,
        "num_heads": 16,
        "logit_scale": 99.998,
    },
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.orig_in21k": {
        "source": "user_defined",
        "org_clip_model_name": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "embedding_dim": 512,
        "visual_feature_dim": 768,
        "num_heads": 12,
        "logit_scale": 85.2323,
        "build_clip_model_func": build_clip_model.build_biomedclip_orig_in21k,
        "get_img_feat_func": get_feat.get_img_feat_hf_hub,
        "get_concept_feat_func": get_feat.get_concept_feat_hf_hub,
    },
}
