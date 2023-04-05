from torch import nn
from model.component.output import MultiModalOutput
from model.component.output import TextTransformerOutput, VisionTransformerOutput
from lavis.models import load_model


class MultiModalBaseModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = load_model(**kwargs)

    def extract_image_features(self, image):
        samples = {'image': image}
        return self.model.extract_features(samples, mode='image').image_features

    def extract_text_features(self, text):
        samples = {'text_input': text}
        return self.model.extract_text_features(samples).text_embeds_proj

    def forward(self, text, image):
        text_features = self.extract_text_features(text)
        image_features = self.extract_image_features(image)

        text_output = TextTransformerOutput(
            last_representation=text_features[0],
            last_layer_output=text_features
        )
        visual_output = VisionTransformerOutput(
            last_representation=image_features[0],
            last_layer_output=image_features
        )
        return MultiModalOutput(
            text_output=text_output,
            visual_output=visual_output
        )
