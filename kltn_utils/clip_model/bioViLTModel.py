import torch.nn as nn


class BioViLTModel(nn.Module):
    def __init__(
        self,
        visual_encoder,
        text_encoder,
    ) -> None:
        super().__init__()

        self.visual_encoder = visual_encoder
        self.text_encoder = text_encoder

    def encode_text(
        self,
        input_ids,
        attention_mask,
    ):
        return self.text_encoder.get_projected_text_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def encode_image(
        self,
        img,
    ):
        return self.visual_encoder(img)
