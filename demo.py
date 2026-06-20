import torch

CLIP_IMAGE_SIZE = {
    "RN50": 224,
    "RN101": 224,
    "RN50x4": 288,
    "RN50x16": 384,
    "RN50x64": 448,
    "ViT-L-14@336px": 336,
    "ViT-B-32": 224,
    "ViT-B-16": 224,
    "ViT-L-14": 224,
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": 224,
    "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K": 224,
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.orig_in21k": 224,
}

CLIP_CONTEXT_LENGTH = {
    "RN50": 77,
    "RN101": 77,
    "RN50x4": 77,
    "RN50x16": 77,
    "RN50x64": 77,
    "ViT-L-14@336px": 77,
    "ViT-B-32": 77,
    "ViT-B-16": 77,
    "ViT-L-14": 77,
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224": 256,
    "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K": 77,
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.orig_in21k": 256,
}


BIOMEDCLIP_NAMES = {
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224.orig_in21k",
}


def _shape_of(x):
    if torch.is_tensor(x):
        return tuple(x.shape)

    if isinstance(x, (list, tuple)):
        return [_shape_of(v) for v in x]

    if isinstance(x, dict):
        return {k: _shape_of(v) for k, v in x.items()}

    return str(type(x))


def _get_device(model):
    return next(model.parameters()).device


def _register_shape_hooks(module, records, prefix=""):
    handles = []

    for name, submodule in module.named_modules():
        if name == "":
            continue

        layer_name = f"{prefix}.{name}" if prefix else name

        def hook_fn(mod, inputs, output, layer_name=layer_name):
            records.append(
                {
                    "layer_name": layer_name,
                    "input_shape": _shape_of(inputs),
                    "output_shape": _shape_of(output),
                }
            )

        handles.append(submodule.register_forward_hook(hook_fn))

    return handles


def get_clip_encoder_shapes(model, model_name, batch_size=2):
    """
    Input:
        model: CLIP model
        model_name: str

    Output:
        visual_layers: list[dict]
        text_layers: list[dict]
    """
    model.eval()
    device = _get_device(model)

    image_size = CLIP_IMAGE_SIZE[model_name]
    context_length = CLIP_CONTEXT_LENGTH[model_name]

    dummy_img = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        device=device,
    )

    dummy_text_token = torch.zeros(
        batch_size,
        context_length,
        dtype=torch.long,
        device=device,
    )

    visual_layers = []
    text_layers = []

    # -------------------------
    # Visual encoder
    # -------------------------
    visual_root = model.visual

    visual_hooks = _register_shape_hooks(
        visual_root,
        visual_layers,
        prefix="visual",
    )

    with torch.no_grad():
        if hasattr(model, "encode_image"):
            _ = model.encode_image(dummy_img)
        else:
            _ = model.visual(dummy_img)

    for h in visual_hooks:
        h.remove()

    # -------------------------
    # Text encoder
    # -------------------------
    if model_name in BIOMEDCLIP_NAMES:
        text_root = model.text
        text_hooks = _register_shape_hooks(
            text_root,
            text_layers,
            prefix="text",
        )

        with torch.no_grad():
            if hasattr(model, "encode_text"):
                _ = model.encode_text(dummy_text_token)
            else:
                _ = model.text(dummy_text_token)

    else:
        # OpenAI CLIP / OpenCLIP text encoder
        text_hooks = []

        if hasattr(model, "token_embedding"):
            text_hooks += _register_shape_hooks(
                model.token_embedding,
                text_layers,
                prefix="token_embedding",
            )

        if hasattr(model, "transformer"):
            text_hooks += _register_shape_hooks(
                model.transformer,
                text_layers,
                prefix="transformer",
            )

        if hasattr(model, "ln_final"):
            text_hooks += _register_shape_hooks(
                model.ln_final,
                text_layers,
                prefix="ln_final",
            )

        with torch.no_grad():
            _ = model.encode_text(dummy_text_token)

    for h in text_hooks:
        h.remove()

    return visual_layers, text_layers


visual_layers, text_layers = get_clip_encoder_shapes(
    model,
    model_name="RN50",
    batch_size=2,
)

print(visual_layers[0])
print(text_layers[0])
