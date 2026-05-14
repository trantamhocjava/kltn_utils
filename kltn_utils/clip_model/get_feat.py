def get_img_feat_openai(clip_model, img):
    return clip_model.encode_image(img)


def get_img_feat_hf_hub(clip_model, img):
    return clip_model(img, None)[0]


def get_concept_feat_openai(clip_model, concept_token):
    return clip_model.encode_text(concept_token)


def get_concept_feat_hf_hub(clip_model, concept_token):
    return clip_model(None, concept_token)[1]
