def get_pmc_clip_text_embedding(
    model,
    tokenizer,
    texts,
):
    device = next(model.parameters()).device
    model.eval()

    # PMC-CLIP sử dụng context_length = 77
    max_length = model.context_length

    text_inputs = tokenizer(
        list(texts),
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

    with torch.inference_mode():
        # PubMedBERT output:
        # last_hidden_state: (B, 77, 768)
        text_outputs = model.text_encoder(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            token_type_ids=text_inputs.get(
                "token_type_ids",
                None,
            ),
            return_dict=True,
        )

        last_hidden_state = text_outputs.last_hidden_state

        # Token đầu tiên là [CLS]
        # Shape: (B, 768)
        cls_embedding = last_hidden_state[:, 0, :]

        # Chiếu sang không gian embedding chung của PMC-CLIP
        # (B, 768) @ (768, 768) -> (B, 768)
        text_embedding = cls_embedding @ model.text_projection

        # Giống bước chuẩn hóa trong forward chính thức
        text_embedding = F.normalize(
            text_embedding,
            dim=-1,
        )

    return text_embedding
