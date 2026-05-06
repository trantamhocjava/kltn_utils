CLIP_MODEL_FROM_OPENAI = ["ViT-B/32", "ViT-B/16", "ViT-L/14"]
CLIP_MODEL_FROM_HF_HUB = (
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
)


def get_img_feat_from_clip_model(clip_model, clip_model_name, img):
    clip_model.eval()
    with torch.no_grad():
        if clip_model_name in CLIP_MODEL_FROM_OPENAI:
            img_feat = clip_model.encode_image(img)
        elif clip_model_name in CLIP_MODEL_FROM_HF_HUB:
            img_feat = clip_model(img, None)[0]

    return img_feat


def build_clip_model(clip_model_name):
    if clip_model_name in CLIP_MODEL_FROM_OPENAI:
        model, _ = clip.load(clip_model_name)
        tokenizer = clip.tokenize
    elif clip_model_name in CLIP_MODEL_FROM_HF_HUB:
        model, _ = create_model_from_pretrained(clip_model_name)
        tokenizer = get_tokenizer(clip_model_name)

    return model, tokenizer


clip_model = build_clip_model("ViT-B/32")
img_feat = get_img_feat_from_clip_model(clip_model, "ViT-B/32", img)


MODELS = {
    "Salesforce/blip-image-captioning-base": {
        "num_param": 247444600,
        "load_model_time_second": 11.204865455627441,
        "infer_1_img_second": 2.57,
    },
    ...
}
