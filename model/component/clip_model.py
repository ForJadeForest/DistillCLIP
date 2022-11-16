from typing import Dict

from torch import nn

from .output import ControlOutput
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder


class CLIPModel(nn.Module):
    def __init__(self, is_student, vit_paras: Dict, text_encoder_para: Dict, tea_transformer_width=None):
        super().__init__()
        self.image_encoder = ImageEncoder(is_student=is_student, vit_paras=vit_paras,
                                          tea_transformer_width=tea_transformer_width)
        self.text_encoder = TextEncoder(is_student=is_student, **text_encoder_para,
                                        tea_transformer_width=tea_transformer_width)
        self.arch_para = vit_paras.update(text_encoder_para)
        self.is_student = is_student
        self.img_layers = vit_paras['layers']
        self.text_layers = text_encoder_para['transformer_layers']

    def encode_image(self, image, control_output: ControlOutput, only_last_state=True):
        return self.image_encoder(image, control_output, only_last_state)

    def encode_text(self, text, control_output: ControlOutput, only_last_state=True):
        return self.text_encoder(text, control_output, only_last_state)

    def forward(self, text, image, control_output: ControlOutput, only_last_state=True):
        image_output = self.encode_image(image, control_output, only_last_state)
        text_output = self.encode_text(text, control_output, only_last_state)
        if only_last_state:
            image_feature = image_output / image_output.norm(dim=1, keepdim=True)
            text_feature = text_output / text_output.norm(dim=1, keepdim=True)
            logits = image_feature @ text_feature.t()
            return image_feature, text_feature, logits
        else:
            image_feature = image_output.representations / image_output.representations.norm(dim=1, keepdim=True)
            text_feature = text_output.representations / text_output.representations.norm(dim=1, keepdim=True)
            logits = image_feature @ text_feature.t()
            return image_output, text_output, logits

    def init_layers_with_teacher(self, text_layer_map, image_layer_map, teacher_state_dict=None, init_type=None):
        self.image_encoder.init_layers_with_teacher(image_layer_map, teacher_state_dict, init_type)
        self.text_encoder.init_layers_with_teacher(text_layer_map, teacher_state_dict, init_type)

    def hyper_para(self):
        return self.arch_para


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
