import torch
from torch import nn
from model.component.output import MultiModalOutput
from lavis.models import load_model, CLIP


class MultiModalBaseModel(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        self.model = load_model(device=device, **kwargs)

    @torch.no_grad()
    def encode_image(self, image):
        raise NotImplemented

    @torch.no_grad()
    def encode_text(self, text):
        raise NotImplemented

    @torch.no_grad()
    def forward(self, text, image):
        text_features = self.encode_text(text)
        image_features = self.encode_image(image)

        return MultiModalOutput(
            text_output=text_features,
            visual_output=image_features
        )
