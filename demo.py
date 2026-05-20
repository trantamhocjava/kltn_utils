def get_txt_feat(texts, clip_model, clip_model_name, tokenizer, batch_size):
    text_token = tokenizer(texts)

    text_loader = DataLoader(
        TensorDataset(text_token),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    res_txt_feat = []

    clip_model.cuda()
    clip_model.eval()
    for batch in text_loader:
        text_token = batch[0].cuda()
        txt_feat = get_concept_feat_from_clip_model(
            clip_model, clip_model_name, text_token
        ).cpu()

        res_txt_feat.append(txt_feat)

    res_txt_feat = torch.cat(res_txt_feat, dim=0)

    return res_txt_feat
