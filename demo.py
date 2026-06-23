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


text_inputs = tokenizer(
    text=["a chest x-ray image"],
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

model.cuda()
model.eval()
for epoch_idx, batch in enumerate(text_loader):
    rank_zero_info_newline(f"run batch {epoch_idx + 1} / {len(text_loader)}")

    input_ids, attention_mask = batch
    input_ids = input_ids.cuda()
    attention_mask = attention_mask.cuda()

    txt_feat = get_pubmedclip_text_feat(model, input_ids, attention_mask).cpu()

    res_txt_feat.append(txt_feat)

res_txt_feat = torch.cat(res_txt_feat, dim=0)


with torch.no_grad():
    text_feat = get_pubmedclip_text_feat(
        model,
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
    )
