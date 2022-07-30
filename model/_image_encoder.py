from torch import nn

try:
    from ._common import VisionTransformer
    from ._utils import output_filter
except ModuleNotFoundError:
    from _common import VisionTransformer
    from _utils import output_filter


class ImageEncoder(nn.Module):
    def __init__(self, is_student, vit_paras, tea_transformer_width=None):
        super().__init__()
        self.vit_paras = vit_paras

        self.visual = VisionTransformer(**vit_paras)
        self.is_student = is_student
        self.embedding_projection = None
        self.hidden_projection = None
        self.layers = vit_paras['layers']
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

    def encode_image(self, image, only_last_state=True):
        last_state, attention_maps, representations, embedding = self.visual(image)

        if only_last_state:
            return last_state

        return output_filter(self.is_student, representations, self.embedding_projection, embedding, attention_maps,
                             last_state, self.hidden_projection)

    def forward(self, image, only_last_state=True):
        return self.encode_image(image, only_last_state)

    @property
    def hyper_para(self):
        return self.vit_paras
