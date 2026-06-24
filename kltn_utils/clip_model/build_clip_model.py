import open_clip
import timm
import torch.nn as nn
from health_multimodal.image.model.pretrained import (
    get_biovil_t_image_encoder,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    CLIPModel,
    CLIPProcessor,
)

from .. import kltn_const, kltn_utils


def build_biomedclip_orig_in21k(clip_model_name):
    org_clip_model_name = kltn_const.CLIP_MODELS[clip_model_name]["org_clip_model_name"]

    model, _ = open_clip.create_model_from_pretrained(org_clip_model_name)
    tokenizer = open_clip.get_tokenizer(org_clip_model_name)

    kltn_utils.rank_zero_info_newline("replace visual encoder weight")
    src_model = timm.create_model(
        "vit_base_patch16_224.orig_in21k",
        pretrained=True,
    )
    src_model.head = nn.Identity()

    model.visual.trunk.load_state_dict(src_model.state_dict())

    return model, tokenizer


def build_pubmed_clip(clip_model_name):
    model = CLIPModel.from_pretrained(clip_model_name)
    tokenizer = CLIPProcessor.from_pretrained(clip_model_name)

    return model, tokenizer


class BioViLTModel(nn.Module):
    def __init__(
        self,
        visual_encoder,
        text_encoder,
    ) -> None:
        super().__init__()

        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder

    def encode_text(
        self,
        input_ids,
        attention_mask,
    ):
        return self.text_encoder.get_projected_text_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def encode_image(
        self,
        pixel_values,
    ):
        return self.visual_encoder(pixel_values)


def build_biovil_t(
    clip_model_name,
):
    visual_encoder = get_biovil_t_image_encoder()

    text_encoder = AutoModel.from_pretrained(
        clip_model_name,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        clip_model_name,
        trust_remote_code=True,
    )

    model = BioViLTModel(
        visual_encoder=visual_encoder,
        text_encoder=text_encoder,
    )

    return model, tokenizer
