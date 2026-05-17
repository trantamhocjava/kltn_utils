import copy
import json
import os
from types import SimpleNamespace

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


def save_dict_to_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json_to_dict(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def deepcopy_obj(obj):
    return copy.deepcopy(obj)


def add_prefix_in_dict(data, mode):
    return {f"{mode}_{key}": value for key, value in data.items()}


def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(
            **{key: dict_to_namespace(value) for key, value in obj.items()}
        )

    if isinstance(obj, list):
        return [dict_to_namespace(item) for item in obj]

    return obj


def read_json_to_namespace(file_path):
    result = dict_to_namespace(read_json_to_dict(file_path))
    return result


def save_list_dict_to_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def dict2device(dictionary, device):
    return {key: value.to(device) for key, value in dictionary.items()}


def detach_dict(dictionary):
    return {key: value.detach().cpu() for key, value in dictionary.items()}


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
        train_transform = v2.Compose(kltn_const.IMAGE_PREPROCESS_LIST)
        val_transform = v2.Compose(kltn_const.IMAGE_PREPROCESS_LIST)

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


def list2tuple_for_dict(dictionary):
    result = {}

    for key, value in dictionary.items():
        if isinstance(value, list):
            result[key] = tuple(value)
        else:
            result[key] = value

    return result


def build_optimizer(params, optimizer_config):
    grad_true_param = filter(lambda p: p.requires_grad, params)

    optimizer_config = vars(optimizer_config)
    optimizer_name = optimizer_config.pop("optimizer")
    optimizer_config = list2tuple_for_dict(optimizer_config)
    optimizer_config["params"] = grad_true_param

    if optimizer_name == "sgd":
        optimizer = optim.SGD(**optimizer_config)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(**optimizer_config)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(**optimizer_config)

    return optimizer


def build_scheduler(optimizer, scheduler_config):
    if scheduler_config is None:
        return None, None

    scheduler_config = vars(scheduler_config)
    scheduler_name = scheduler_config.pop("scheduler")
    scheduler_config = list2tuple_for_dict(scheduler_config)
    scheduler_config["optimizer"] = optimizer

    monitor = None
    if scheduler_name == "LinearLR":
        scheduler = optim.lr_scheduler.LinearLR(**scheduler_config)
    elif scheduler_name == "StepLR":
        scheduler = optim.lr_scheduler.StepLR(**scheduler_config)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(**scheduler_config)
        monitor = "val_loss"
    elif scheduler_name == "transformer_lr_scheduler":
        scheduler = get_linear_schedule_with_warmup(**scheduler_config)

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
    elif source == "user_defined":
        build_clip_model_func = kltn_const.CLIP_MODELS[clip_model_name][
            "build_clip_model_func"
        ]
        model, tokenizer = build_clip_model_func(clip_model_name)

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
        elif source == "user_defined":
            get_img_feat_func = kltn_const.CLIP_MODELS[clip_model_name][
                "get_img_feat_func"
            ]
            img_feat = get_img_feat_func(clip_model, img)

    return img_feat


def get_concept_feat_from_clip_model(clip_model, clip_model_name, concept_token):
    source = kltn_const.CLIP_MODELS[clip_model_name]["source"]

    with torch.no_grad():
        if source == "openai":
            concept_feat = clip_model.encode_text(concept_token)
        elif source == "hf-hub":
            concept_feat = clip_model(None, concept_token)[1]
        elif source == "user_defined":
            get_concept_feat_func = kltn_const.CLIP_MODELS[clip_model_name][
                "get_concept_feat_func"
            ]
            concept_feat = get_concept_feat_func(clip_model, concept_token)

    return concept_feat


def is_data_type(variable, data_type):
    result = None

    if data_type == "float":
        result = isinstance(variable, (np.floating, float))

    return result


def update_optimizer(optimizer):
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def cal_label_accuracy(y_true, y_pred, mode):
    if mode == "acc":
        result = metrics.accuracy_score(y_true, y_pred) * 100
    elif mode == "bmac":
        result = metrics.balanced_accuracy_score(y_true, y_pred) * 100

    return result


def cal_concept_accuracy(c_true, c_pred, mode):
    if mode == "acc":
        result = (c_true == c_pred).mean() * 100
    elif mode == "overall_acc":
        result = np.mean(np.all(c_true == c_pred, axis=1)) * 100

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


def get_class2concept_matrix(concept2class):
    num_concept = len(concept2class)
    num_class = len(set(concept2class))

    concept2class = torch.tensor(concept2class).long().view(1, -1)
    matrix = torch.zeros(num_class, num_concept)
    matrix.scatter_(0, concept2class, 1)

    return matrix


def build_class_concept_matrix(concept2class, num_class):
    num_concept = len(concept2class)

    matrix = torch.zeros(num_class, num_concept, dtype=torch.long)

    for concept_idx, class_indices in enumerate(concept2class):
        for class_idx in class_indices:
            matrix[class_idx, concept_idx] = 1

    return matrix


def load_img_classify_data(dataset_dir, class_names):
    file_paths = []
    labels = []
    for class_index, class_name in enumerate(class_names):
        file_paths = [
            f"{dataset_dir}/{class_name}/{i}"
            for i in os.listdir(f"{dataset_dir}/{class_name}")
        ]
        file_paths += file_paths
        labels += [class_index] * len(file_paths)

    return file_paths, labels


def get_sublist(src_list, select_idx):
    return [src_list[idx] for idx in select_idx]
