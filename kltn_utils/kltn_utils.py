import csv
import json
import os

import clip
import numpy as np
import timm
import torch
import torch.distributed as dist
from open_clip import create_model_from_pretrained, get_tokenizer
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info
from torch import optim
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2
from transformers import get_linear_schedule_with_warmup

from . import kltn_const


def rank_zero_info_newline(text):
    rank_zero_info(f"\n{text}\n")


def seed_everything_in_pl():
    seed_everything(kltn_const.SEEDING, workers=True)
    torch.use_deterministic_algorithms(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_mode(monitor):
    if monitor in kltn_const.METRIC_MAX:
        return "max"
    else:
        return "min"


def destroy_process_group():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def create_csv_file(file_path, columns):
    if os.path.exists(file_path):
        return

    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)


def fill_1line_in_csv_file(file_path, line):
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(line)


def log_in_csv(test_result, columns, file_path):
    line = [test_result[column] for column in columns]
    fill_1line_in_csv_file(file_path, line)


def save_dict_to_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json_to_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def build_transform(config):
    """Uniform preprocess follow preprocess architecture got from the code below
    ```
    from open_clip import create_model_from_pretrained, get_tokenizer

    _, preprocess = create_model_from_pretrained(
        "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    )
    ```

    """
    if config.transform == "uniform":
        train_transform = v2.Compose(kltn_const.PREPROCESS_LIST)
        val_transform = v2.Compose(kltn_const.PREPROCESS_LIST)

    return train_transform, val_transform


def build_blackbox_model(config):
    model = timm.create_model(
        config.model, pretrained=True, num_classes=len(config.class_names)
    )

    return model


def read_img(img_path):
    res = None

    try:
        res = read_image(
            img_path,
            mode=ImageReadMode.RGB,
        )
    except Exception:
        img = Image.open(img_path).convert("RGB")
        res = torch.from_numpy(np.array(img, dtype=np.uint8)).permute(2, 0, 1)

    return res


def build_optimizer(model, config):
    grad_true_param = filter(lambda p: p.requires_grad, model.parameters())

    if config.optimizer == "sgd":
        optimizer = optim.SGD(
            grad_true_param,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "sgd_v1":
        optimizer = optim.SGD(
            grad_true_param,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            grad_true_param,
            lr=config.lr,
            betas=(config.beta_1, config.beta_2),
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adam_v1":
        optimizer = optim.Adam(
            grad_true_param,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            grad_true_param,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    return optimizer


def build_scheduler(optimizer, config):
    if config.scheduler is None:
        return None, None

    monitor = None
    if config.scheduler == "LinearLR":
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1,
            end_factor=0.01,
            total_iters=config.epochs,
        )
    elif config.scheduler == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.decrease_every,
            gamma=1 / config.lr_divisor,
        )
    elif config.scheduler == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        monitor = "val_loss"
    elif config.scheduler == "transformer_lr_scheduler":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.epochs * config.n_batchs,
        )

    return scheduler, monitor


def build_clip_model(clip_model_name):
    if clip_model_name in kltn_const.CLIP_MODEL_FROM_OPENAI:
        model, _ = clip.load(clip_model_name)
        tokenizer = clip.tokenize
    elif clip_model_name in kltn_const.CLIP_MODEL_FROM_HF_HUB:
        model, _ = create_model_from_pretrained(clip_model_name)
        tokenizer = get_tokenizer(clip_model_name)

    return model, tokenizer


def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    for param in m.parameters():
        param.requires_grad = True


def get_img_feat_from_clip_model(clip_model, clip_model_name, img):
    if clip_model_name in kltn_const.CLIP_MODEL_FROM_OPENAI:
        img_feat = clip_model.encode_image(img)
    elif clip_model_name in kltn_const.CLIP_MODEL_FROM_HF_HUB:
        img_feat = clip_model(img, None)[0]

    return img_feat


def get_concept_feat_from_clip_model(clip_model, clip_model_name, concept_token):
    if clip_model_name in kltn_const.CLIP_MODEL_FROM_OPENAI:
        concept_feat = clip_model.encode_text(concept_token)
    elif clip_model_name in kltn_const.CLIP_MODEL_FROM_HF_HUB:
        concept_feat = clip_model(None, concept_token)[1]

    return concept_feat
