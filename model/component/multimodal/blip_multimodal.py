import torch
from model.component.output import TextTransformerOutput, VisionTransformerOutput
from .base_multimodal_model import MultiModalBaseModel


class BLIPMultiModal(MultiModalBaseModel):
    def __init__(self,
                 device,
                 name='blip_feature_extractor',
                 model_type='base',
                 is_eval=True,
                 **kwargs):
        super().__init__(device=device,
                         name=name,
                         model_type=model_type,
                         is_eval=is_eval,
                         **kwargs)

    @torch.no_grad()
    def encode_image(self, image):
        image_embeds = self.model.visual_encoder.forward_features(image)
        image_features = self.model.vision_proj(image_embeds[:, 0, :])
        return VisionTransformerOutput(last_representation=image_features)

    @torch.no_grad()
    def encode_text(self, text):
        if isinstance(text, list):
            text = self.model.tokenizer(text, return_tensors="pt", padding=True).to(self.model.device)
        text_output = self.model.text_encoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state

        text_features = self.model.text_proj(text_embeds[:, 0, :])
        return TextTransformerOutput(last_representation=text_features)
