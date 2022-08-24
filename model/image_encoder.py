from torch import nn

try:
    from _common import VisionTransformer, VisionTransformerFFT
    from _utils import output_filter, load

except ModuleNotFoundError:
    from ._common import VisionTransformer, VisionTransformerFFT
    from ._utils import output_filter, load


class ImageEncoderFFT(nn.Module):
    def __init__(self, is_student, vit_paras):
        super().__init__()
        self.vit_paras = vit_paras
        self.visual = VisionTransformerFFT(**vit_paras)
        self.is_student = is_student
        self.layers = vit_paras['layers']

    def encode_image(self, image):
        last_state = self.visual(image)
        return last_state

    def forward(self, image):
        return self.encode_image(image)

    def hyper_para(self):
        return self.vit_paras


class ImageEncoder(nn.Module):
    def __init__(self, is_student, vit_paras, tea_transformer_width=None, drop_out=0.1):
        super().__init__()
        self.vit_paras = vit_paras
        self.visual = VisionTransformer(**vit_paras, drop_out=drop_out)
        self.is_student = is_student
        self.embedding_projection = None
        self.hidden_projection = None
        self.layers = vit_paras['layers']
        self.no_trans = False
        if self.vit_paras['width'] == tea_transformer_width:
            self.no_trans = True
        if is_student:
            self.embedding_projection = nn.Linear(vit_paras['width'], tea_transformer_width)
            self.hidden_projection = nn.Linear(vit_paras['width'], tea_transformer_width)
        self.initialize_parameters()

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

    def encode_image(self, image, only_last_state=True, need_attn_score=False, need_value_map=False,
                     need_attn_prob=False, need_rep=False, need_emb=False):
        last_state, attention_maps, representations, embedding, attention_probs, value_map = self.visual(image,
                                                                                                         need_attn_score,
                                                                                                         need_value_map,
                                                                                                         need_attn_prob,
                                                                                                         need_rep)
        if only_last_state:
            return last_state

        return output_filter(self.is_student, representations, self.embedding_projection, embedding, attention_maps,
                             last_state, self.hidden_projection, attention_probs, value_map, need_emb, need_rep,
                             need_attn_score, self.no_trans)

    def forward(self, image, only_last_state=True, **need_para):
        return self.encode_image(image, only_last_state, **need_para)

    def init_layers_with_teacher(self, layer_map, teacher_state_dict=None, init_type=None):
        import re
        pattern = re.compile('visual.transformer.resblocks.([\d])')
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
                tea_key = re.sub(re.compile('\d'), map_layer(int(res[0])), string=key, count=1)
                my_model_state_dict[key] = tea_state_dict[tea_key]
        self.visual.load_state_dict(my_model_state_dict)
        print('init with teacher weight success!')

    def hyper_para(self):
        return self.vit_paras
