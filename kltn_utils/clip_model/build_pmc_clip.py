from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

import torch
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from kltn_utils.third_party.pmc_clip import (
    create_model_and_transforms,
)
from kltn_utils.third_party.pmc_clip.pretrained import (
    download_pretrained,
)

PMC_CLIP_CHECKPOINT_URL = (
    "https://huggingface.co/datasets/" "axiong/pmc_oa/resolve/main/checkpoint.pt"
)

PMC_CLIP_TOKENIZER_NAME = (
    "microsoft/" "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
)


def _extract_state_dict(checkpoint):
    """
    Lấy state_dict từ checkpoint PMC-CLIP.
    """
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    return checkpoint


def _clean_pmc_clip_state_dict(state_dict):
    """
    Chuẩn hóa state_dict của PMC-CLIP.

    Xử lý:
    - prefix `module.` sinh ra bởi DistributedDataParallel;
    - buffer `position_ids` của phiên bản Transformers cũ.
    """
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        # Checkpoint được train bằng DDP có thể chứa prefix module.
        key = key.removeprefix("module.")

        # Transformers mới không lưu position_ids trong state_dict.
        if key.endswith("embeddings.position_ids"):
            continue

        cleaned_state_dict[key] = value

    return cleaned_state_dict


def build_pmc_clip_model(clip_model_name) -> Tuple[
    nn.Module,
    PreTrainedTokenizerBase,
]:
    print("Load model PMC-CLIP")

    # =====================================================
    # 1. Device
    # =====================================================
    device = "cpu"

    # =====================================================
    # 2. Cấu hình PMC-CLIP
    # =====================================================
    args = SimpleNamespace(
        model="RN50_fusion4",
        pretrained="",
        hugging_face=True,
        mlm=True,
        crop_scale=0.1,
        device=device,
    )

    # =====================================================
    # 3. Tạo kiến trúc model
    # =====================================================
    model, _, _ = create_model_and_transforms(
        args=args,
        precision="fp32",
        device=device,
    )

    # =====================================================
    # 4. Tải checkpoint
    # =====================================================
    checkpoint_root = Path.home() / ".cache" / "pmc_clip"

    checkpoint_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = download_pretrained(
        url=PMC_CLIP_CHECKPOINT_URL,
        root=str(checkpoint_root),
    )

    # =====================================================
    # 5. Load checkpoint
    #
    # PMC-CLIP checkpoint chứa object NumPy nên với
    # PyTorch >= 2.6 phải dùng weights_only=False.
    # Chỉ thực hiện vì đây là checkpoint chính thức.
    # =====================================================
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = _extract_state_dict(checkpoint)

    if not isinstance(state_dict, dict):
        raise TypeError(
            "PMC-CLIP checkpoint không chứa state_dict hợp lệ. "
            f"Nhận được kiểu: {type(state_dict)}"
        )

    state_dict = _clean_pmc_clip_state_dict(state_dict)

    incompatible_keys = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if incompatible_keys.missing_keys:
        raise RuntimeError(
            "Thiếu trọng số khi load PMC-CLIP: " f"{incompatible_keys.missing_keys}"
        )

    if incompatible_keys.unexpected_keys:
        raise RuntimeError(
            "Có trọng số không mong đợi khi load PMC-CLIP: "
            f"{incompatible_keys.unexpected_keys}"
        )

    model = model.to(device)

    # =====================================================
    # 6. Load tokenizer
    # =====================================================
    tokenizer = AutoTokenizer.from_pretrained(PMC_CLIP_TOKENIZER_NAME)

    return model, tokenizer
