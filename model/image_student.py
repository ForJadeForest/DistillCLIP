from torch import nn
from _common import VisionTransformer


class ImageStudent(nn.Module):
    def __init__(self, vision_width, image_resolution, vision_patch_size, vision_layers, embed_dim):
        super().__init__()
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

    def encode_image(self, image):
        return self.visual(image)

    def forward(self, image):
        return self.encode_image(image)
