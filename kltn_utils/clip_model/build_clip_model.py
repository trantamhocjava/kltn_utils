import open_clip
import timm
import torch.nn as nn

from .. import kltn_const


def build_biomedclip_orig_in21k(clip_model_name):
    org_clip_model_name = kltn_const.CLIP_MODELS[clip_model_name]["org_clip_model_name"]

    model, _ = open_clip.create_model_from_pretrained(org_clip_model_name)
    tokenizer = open_clip.get_tokenizer(org_clip_model_name)

    src_model = timm.create_model(
        "vit_base_patch16_224.orig_in21k",
        pretrained=True,
    )
    src_model.head = nn.Identity()

    model.visual.trunk.load_state_dict(src_model.state_dict())

    return model, tokenizer
