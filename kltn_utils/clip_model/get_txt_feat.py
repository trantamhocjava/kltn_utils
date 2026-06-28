import torch
from torch.utils.data import DataLoader, TensorDataset


def get_pubmedclip_text_feat(model, input_ids, attention_mask=None):
    text_outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    pooled_text_feat = text_outputs.pooler_output
    # shape: (B, 512)

    text_feat = model.text_projection(pooled_text_feat)
    # shape: (B, 512)

    return text_feat


def get_txt_feat_pubmedclip_from_texts(texts, clip_model, tokenizer, batch_size):
    text_inputs = tokenizer(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    text_loader = DataLoader(
        TensorDataset(text_inputs["input_ids"], text_inputs["attention_mask"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    res_txt_feat = []

    clip_model.cuda()
    clip_model.eval()
    with torch.no_grad():
        for epoch_idx, batch in enumerate(text_loader):
            print(f"run batch {epoch_idx + 1} / {len(text_loader)}")

            input_ids, attention_mask = batch
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

            txt_feat = get_pubmedclip_text_feat(
                clip_model, input_ids, attention_mask
            ).cpu()

            res_txt_feat.append(txt_feat)

    txt_feat = torch.cat(res_txt_feat, dim=0)

    return txt_feat


def get_txt_feat_BiomedVLP_from_texts(texts, clip_model, tokenizer, batch_size):
    text_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    text_loader = DataLoader(
        TensorDataset(text_input["input_ids"], text_input["attention_mask"]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    res_txt_feat = []

    clip_model.cuda()
    clip_model.eval()
    with torch.no_grad():
        for epoch_idx, batch in enumerate(text_loader):
            print(f"run batch {epoch_idx + 1} / {len(text_loader)}")

            input_ids, attention_mask = batch
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

            txt_feat = clip_model.encode_text(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).cpu()

            res_txt_feat.append(txt_feat)

    txt_feat = torch.cat(res_txt_feat, dim=0)

    return txt_feat


def get_txt_feat_PMC_CLIP_from_texts(texts, clip_model, tokenizer, batch_size):
    # PMC-CLIP sử dụng context_length = 77
    max_length = clip_model.context_length

    text_inputs = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    text_loader = DataLoader(
        TensorDataset(
            text_inputs["input_ids"],
            text_inputs["attention_mask"],
            text_inputs["token_type_ids"],
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    res_txt_feat = []

    clip_model.cuda()
    clip_model.eval()
    with torch.no_grad():
        for epoch_idx, batch in enumerate(text_loader):
            print(f"run batch {epoch_idx + 1} / {len(text_loader)}")

            input_ids, attention_mask, token_type_ids = batch
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
            token_type_ids = token_type_ids.cuda()

            text_outputs = clip_model.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )

            last_hidden_state = text_outputs.last_hidden_state

            # Token đầu tiên là [CLS]
            # Shape: (B, 768)
            cls_embedding = last_hidden_state[:, 0, :]

            # Chiếu sang không gian embedding chung của PMC-CLIP
            # (B, 768) @ (768, 768) -> (B, 768)
            txt_feat = cls_embedding @ clip_model.text_projection

            res_txt_feat.append(txt_feat)

    txt_feat = torch.cat(res_txt_feat, dim=0)

    return txt_feat
