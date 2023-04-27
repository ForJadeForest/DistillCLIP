import torch.nn.functional as f
from typing import Optional

from torch import nn

from .output import ControlOutput, CLIPOutput


class CLIPModel(nn.Module):
    def __init__(self, is_student: bool, image_encoder: nn.Module, text_encoder: nn.Module,
                 norm=False, only_last_rep=False, only_rep=False):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.is_student = is_student
        self.norm = norm
        self.only_last_rep = only_last_rep
        self.only_rep = only_rep

    def encode_image(self, image, control_output: ControlOutput = None):
        if control_output is None:
            control_output = ControlOutput()
        if self.only_last_rep:
            return self.image_encoder(image, control_output).last_representation
        return self.image_encoder(image, control_output)

    def encode_text(self, text, control_output: ControlOutput = None):
        if control_output is None:
            control_output = ControlOutput()
        if isinstance(text, list):
            from clip import tokenize
            text = tokenize(text, truncate=True).to(next(self.image_encoder.parameters()).device)
        if self.only_last_rep:
            return self.text_encoder(text, control_output).last_representation
        return self.text_encoder(text, control_output)

    def forward(self, text, image, control_output: Optional[ControlOutput] = None):
        if control_output is None:
            control_output = ControlOutput()

        image_output = self.encode_image(image, control_output)
        text_output = self.encode_text(text, control_output)
        if not self.only_last_rep:
            image_output.last_representation = f.normalize(image_output.last_representation, p=2, dim=-1)
            text_output.last_representation = f.normalize(text_output.last_representation, p=2, dim=-1)

            return CLIPOutput(visual_output=image_output, text_output=text_output)
        else:
            image_feature = f.normalize(image_output, p=2, dim=-1)
            text_feature = f.normalize(text_output, p=2, dim=-1)

            logits = image_feature @ text_feature.t()
            return image_feature, text_feature, logits

    def init_layers_with_teacher(self, text_layer_map, image_layer_map, teacher_state_dict=None, init_type=None):
        self.image_encoder.init_layers_with_teacher(image_layer_map, teacher_state_dict, init_type)
        self.text_encoder.init_layers_with_teacher(text_layer_map, teacher_state_dict, init_type)

    def hyper_para(self):
        visual_para = self.image_encoder.hyper_para()
        text_para = self.text_encoder.hyper_para()
        res = {}
        for k, v in visual_para.items():
            res['image_' + k] = v
        for k, v in text_para.items():
            res['text_' + k] = v
        return res


if __name__ == '__main__':
    trans_para = {
        'embed_dim': 512,
        'context_length': 77,
        'vocab_size': 49408,
        'transformer_width': 512,
        'transformer_heads': 8,
        'transformer_layers': 4,
    }
    vit_para = {
        'input_resolution': 224,
        'patch_size': 32,
        'width': 768,
        'layers': 4,
        'heads': 12,
        'output_dim': 512,
        'drop_out': 0.1
    }
    m = CLIPModel(False, vit_para, trans_para)
    print(m.state_dict())
