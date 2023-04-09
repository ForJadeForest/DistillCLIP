from torch import nn
from model.component.output import MultiModalOutput
from model.component.output import TextTransformerOutput, VisionTransformerOutput
from lavis.models import load_model
from lavis import BlipFeatureExtractor, CLIP

class MultiModalBaseModel(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.model = load_model(device=device, **kwargs)

    def encode_image(self, image):
        samples = {'image': image}
        try:
            image_feature = self.model.extract_features(samples, mode='image').image_embeds_proj[:, 0, :]
        except:
            image_feature = self.model.extract_features(samples)
        return VisionTransformerOutput(last_representation=image_feature)

    def encode_text(self, text):
        samples = {'text_input': text}
        try:
            text_feature = self.model.extract_features(samples, mode='text').text_embeds_proj[:, 0, :]
        except:
            text_feature = self.model.extract_features(samples)
        return TextTransformerOutput(last_representation=text_feature)

    def forward(self, text, image):
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)
        i2t_logits = image_features.last_representation @ text_features.last_representation.t()
        t2i_logits = text_features.last_representation @ image_features.last_representation.t()

        return MultiModalOutput(
            text_output=text_features,
            visual_output=image_features,
            i2t_logits=i2t_logits,
            t2i_logits=t2i_logits
        )
