import csv
import gzip
import json
import os
import shutil

import numpy as np
import open_clip
import timm
import torch
import torch.distributed as dist
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities import rank_zero_info
from sklearn import metrics
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2
from transformers import get_linear_schedule_with_warmup

from . import kltn_const
from .dataset import ImageDataset


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


def save_list_dict_to_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl_to_list(file_path):
    data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


def build_transform(transform_method):
    """Uniform preprocess follow preprocess architecture got from the code below
    ```
    from open_clip import create_model_from_pretrained, get_tokenizer

    _, preprocess = create_model_from_pretrained(
        "hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    )
    ```

    """
    if transform_method == "uniform":
        train_transform = v2.Compose(kltn_const.PREPROCESS_LIST)
        val_transform = v2.Compose(kltn_const.PREPROCESS_LIST)

    return train_transform, val_transform


def build_blackbox_model(model_name, num_class):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_class)

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
    elif config.optimizer == "adam":
        optimizer = optim.Adam(
            grad_true_param,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
        )
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            grad_true_param,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
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
            optimizer, step_size=config.step_size, gamma=config.gamma
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
    source = kltn_const.CLIP_MODELS[clip_model_name]["source"]

    if source == "openai":
        # Dùng pretrained từ OpenAI qua open-clip-torch
        model, _, _ = open_clip.create_model_and_transforms(
            model_name=clip_model_name,
            pretrained="openai",
        )
        tokenizer = open_clip.get_tokenizer(clip_model_name)

    elif source == "hf-hub":
        model, _ = open_clip.create_model_from_pretrained(clip_model_name)
        tokenizer = open_clip.get_tokenizer(clip_model_name)

    return model, tokenizer


def freeze_module(m):
    for param in m.parameters():
        param.requires_grad = False


def unfreeze_module(m):
    for param in m.parameters():
        param.requires_grad = True


def get_img_feat_from_clip_model(clip_model, clip_model_name, img):
    source = kltn_const.CLIP_MODELS[clip_model_name]["source"]

    with torch.no_grad():
        if source == "openai":
            img_feat = clip_model.encode_image(img)
        elif source == "hf-hub":
            img_feat = clip_model(img, None)[0]

    return img_feat


def get_concept_feat_from_clip_model(clip_model, clip_model_name, concept_token):
    source = kltn_const.CLIP_MODELS[clip_model_name]["source"]

    with torch.no_grad():
        if source == "openai":
            concept_feat = clip_model.encode_text(concept_token)
        elif source == "hf-hub":
            concept_feat = clip_model(None, concept_token)[1]

    return concept_feat


def is_data_type(variable, data_type):
    res = None

    if data_type in kltn_const.DATA_TYPES:
        res = type(variable).__name__ == data_type
    else:
        if data_type == "float":
            res = isinstance(variable, (np.floating, float))

    return res


def uncompress_gzip(src_file_path, dst_file_path):
    with gzip.open(src_file_path, "rb") as f_in:
        with open(dst_file_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    rank_zero_info_newline(f"Uncompress {src_file_path} to {dst_file_path} OK")


def update_optimizer(optimizer):
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def cal_label_accuracy(y_true, y_pred, mode):
    if mode == "acc":
        result = metrics.accuracy_score(y_true, y_pred) * 100
    elif mode == "bmac":
        result = metrics.balanced_accuracy_score(y_true, y_pred) * 100

    return result


def get_img_feat(
    clip_model,
    clip_model_name,
    dataset_dir,
    batch_size,
    transform,
    class_names,
):
    imgset = ImageDataset(
        dataset_dir=dataset_dir,
        transforms=transform,
        class_names=class_names,
    )

    img_loader = DataLoader(
        imgset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    res_img_feat = []
    res_label = []

    clip_model.cuda()
    clip_model.eval()
    for img, label in img_loader:
        img = img.cuda()
        img_feat = get_img_feat_from_clip_model(clip_model, clip_model_name, img).cpu()
        res_img_feat.append(img_feat)
        res_label.append(label)

    res_img_feat = torch.cat(res_img_feat, dim=0)
    res_label = torch.cat(res_label, dim=0)

    return res_img_feat, res_label


def get_txt_feat(texts, clip_model, clip_model_name, tokenizer, batch_size):
    text_loader = DataLoader(
        TensorDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    res_txt_feat = []

    clip_model.cuda()
    clip_model.eval()
    for text in text_loader:
        text_token = tokenizer(text).cuda()

        txt_feat = get_concept_feat_from_clip_model(
            clip_model, clip_model_name, text_token
        ).cpu()

        res_txt_feat.append(txt_feat)

    res_txt_feat = torch.cat(res_txt_feat, dim=0)

    return res_txt_feat


def get_concept2class_matrix(concept2class):
    num_concept = len(concept2class)
    num_class = len(set(concept2class))

    concept2class = torch.tensor(concept2class).long().view(1, -1)
    matrix = torch.zeros(num_class, num_concept)
    matrix.scatter_(0, concept2class, 1)

    return matrix
