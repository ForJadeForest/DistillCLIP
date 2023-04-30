from torch.nn import functional as f


def get_logits(clip_out, logits_scale=None):
    image_features = clip_out.visual_output.last_representation
    text_features = clip_out.text_output.last_representation

    image_features = f.normalize(image_features, p=2, dim=-1)
    text_features = f.normalize(text_features, p=2, dim=-1)

    if logits_scale:
        logits_per_image = logits_scale.exp() * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
    else:
        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
    return logits_per_image, logits_per_text
