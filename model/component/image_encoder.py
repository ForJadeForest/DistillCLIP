import torch
from torch import nn

from ._common import VisionTransformer
from .output import ControlOutput, VisionTransformerOutput


class ImageEncoder(nn.Module):
    def __init__(self, is_student, tea_transformer_width=None, **vit_paras):
        super().__init__()
        self.layers = vit_paras['layers']
        if 'need_layers' not in vit_paras or vit_paras['need_layers'] is None:
            vit_paras['need_layers'] = tuple(range(self.layers))
        self.vit_paras = vit_paras
        self.visual = VisionTransformer(**vit_paras)
        self.is_student = is_student
        self.embedding_projection = None
        self.hidden_projection = None

        self.no_trans = False
        if self.vit_paras['width'] == tea_transformer_width:
            self.no_trans = True
        if is_student:
            self.embedding_projection = nn.Linear(vit_paras['width'], tea_transformer_width)
            self.hidden_projection = nn.Linear(vit_paras['width'], tea_transformer_width)
        self.initialize_parameters()

    @property
    def need_layers(self):
        return self.vit_paras['need_layers']

    def initialize_parameters(self):
        nn.init.normal_(self.visual.class_embedding, std=0.02)
        nn.init.normal_(self.visual.positional_embedding, std=0.01)

        proj_std = (self.visual.transformer.width ** -0.5) * ((2 * self.visual.transformer.layers) ** -0.5)
        attn_std = self.visual.transformer.width ** -0.5
        fc_std = (2 * self.visual.transformer.width) ** -0.5
        for block in self.visual.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.in_proj_bias, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def encode_image(self, image, control_output: ControlOutput):
        vit_output: VisionTransformerOutput = self.visual(image, control_output)
        if self.is_student and not self.no_trans:
            if control_output.need_rep:
                vit_output.representations = [self.hidden_projection(layer_rep) for layer_rep in
                                              vit_output.representations]
            if control_output.need_emb:
                vit_output.embedding = self.embedding_projection(vit_output.embedding)
        if control_output.need_attn_score:
            vit_output.attention_scores = [torch.where(attn_score == float('-inf'),
                                                       torch.zeros_like(attn_score),
                                                       attn_score) for attn_score in vit_output.attention_scores]

        return vit_output

    def forward(self, image, control_output: ControlOutput):
        return self.encode_image(image, control_output)

    def init_layers_with_teacher(self, layer_map, teacher_state_dict=None, init_type=None):
        import re
        pattern = re.compile('visual.transformer.resblocks.(\\d)')
        stu_layer_num = layer_map.stu_total_layer_num
        tea_layer_num = layer_map.tea_total_layer_num
        tea_state_dict = teacher_state_dict
        my_model_state_dict = self.visual.state_dict()
        if init_type is None:
            return
        elif init_type == 'begin':
            map_layer = lambda x: str(x)
        elif init_type == 'end':
            map_layer = lambda x: str(tea_layer_num - stu_layer_num + x)
        elif init_type == 'mid':
            map_layer = lambda x: str(x * layer_map.step)
        else:
            raise ValueError('the init_type should be begin, end, and mid, but got {}'.format(self.init_type))
        for key in my_model_state_dict.keys():
            res = re.findall(pattern, key)
            if key not in tea_state_dict:
                continue
            if not res:
                my_model_state_dict[key] = tea_state_dict[key]
            else:
                tea_key = re.sub(re.compile('\\d'), map_layer(int(res[0])), string=key, count=1)
                my_model_state_dict[key] = tea_state_dict[tea_key]
        self.visual.load_state_dict(my_model_state_dict)
        print('init with teacher weight success!')

    def hyper_para(self):
        return self.vit_paras
