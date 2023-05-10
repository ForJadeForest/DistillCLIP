import torch
from model.component.output import TextTransformerOutput, VisionTransformerOutput
from .base_multimodal_model import MultiModalBaseModel


class CLIPMultiModal(MultiModalBaseModel):
    def __init__(self,
                 device,
                 name='clip',
                 model_type='ViT-B-32',
                 is_eval=True,
                 **kwargs):
        super().__init__(device=device,
                         name=name,
                         model_type=model_type,
                         is_eval=is_eval,
                         **kwargs)

    @torch.no_grad()
    def encode_image(self, image):
        image_feature = self.model.encode_image(image)
        return VisionTransformerOutput(last_representation=image_feature)

    @torch.no_grad()
    def encode_text(self, text):
        if isinstance(text, list):
            text = self.model.tokenizer(text).to(self.model.device)
        text_feature = self.model.encode_text(text)
        return TextTransformerOutput(last_representation=text_feature)
