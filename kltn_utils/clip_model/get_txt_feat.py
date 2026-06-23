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
