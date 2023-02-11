import torch
from torch import nn

from ._common import Transformer, LayerNorm
from .output import ControlOutput, TransformerOutput, TextTransformerOutput


class TextEncoder(nn.Module):
    def __init__(self, transformer_width, transformer_layers, transformer_heads, context_length, need_layers,
                 vocab_size, embed_dim, tea_transformer_width=None, is_student=True, drop_out=0.,
                 compression_embedding=False, embedding_compression_dim=256):
        super().__init__()
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.layers = transformer_layers

        if compression_embedding:
            self.token_embedding = nn.ModuleList([
                nn.Embedding(vocab_size, embedding_compression_dim),
                nn.Linear(embedding_compression_dim, embed_dim)])
        else:
            self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.transformer_para = dict(
            width=self.transformer_width,
            layers=self.layers,
            heads=self.transformer_heads,
            drop_out=drop_out,
            attn_mask=self.build_attention_mask(),
            need_layers=need_layers
        )
        self.transformer = Transformer(**self.transformer_para)

        self.embedding_projection = None
        self.hidden_projection = None
        self.is_student = is_student
        self.no_trans = False
        if transformer_layers == tea_transformer_width:
            self.no_trans = True
        if is_student:
            self.embedding_projection = nn.Linear(transformer_width, tea_transformer_width)
            self.hidden_projection = nn.Linear(transformer_width, tea_transformer_width)
        self.initialize_parameters()

    @property
    def need_layers(self):
        return self.transformer_para['need_layers']

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text, control_output: ControlOutput = None):
        if control_output is None:
            control_output = ControlOutput()
        embedding = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = embedding + self.positional_embedding
        embedding_res = x
        transformer_output: TransformerOutput = self.transformer(x, control_output)
        x = self.ln_final(transformer_output.last_layer_output)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        last_layer_output = x @ self.text_projection
        # x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        if self.is_student and not self.no_trans:
            if control_output.need_rep:
                transformer_output.representations = [self.hidden_projection(layer_rep) for layer_rep in
                                                      transformer_output.representations]
            if control_output.need_emb:
                embedding_res = self.embedding_projection(embedding_res)
        if control_output.need_attn_score:
            transformer_output.attention_scores = [torch.where(attn_score == float('-inf'),
                                                               torch.zeros_like(attn_score),
                                                               attn_score) for attn_score in
                                                   transformer_output.attention_scores]
        return TextTransformerOutput(last_representation=last_layer_output[torch.arange(x.shape[0]), text.argmax(dim=-1)],
                                     last_layer_output=last_layer_output,
                                     attention_scores=transformer_output.attention_scores,
                                     attention_probs=transformer_output.attention_probs,
                                     representations=transformer_output.representations,
                                     value_map=transformer_output.value_map,
                                     embedding=embedding_res)

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.in_proj_bias, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def forward(self, text, control_output: ControlOutput):
        return self.encode_text(text, control_output)

    def hyper_para(self):
        return {
            'context_length': self.context_length,
            'transformer_width': self.transformer_width,
            'transformer_layers': self.layers,
            'transformer_heads': self.transformer_heads,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
        }

    def init_layers_with_teacher(self, layer_map, teacher_state_dict=None, init_type=None):
        if init_type is None:
            return
        import re
        pattern = re.compile('transformer.resblocks.([\d])')
        stu_layer_num = layer_map.stu_total_layer_num
        tea_layer_num = layer_map.tea_total_layer_num
        tea_state_dict = teacher_state_dict
        my_model_state_dict = self.state_dict()

        if init_type == 'begin':
            map_layer = lambda x: str(x)
        elif init_type == 'end':
            map_layer = lambda x: str(tea_layer_num - stu_layer_num + x)
        elif init_type == 'mid':
            map_layer = lambda x: str(x * layer_map.step)
        else:
            raise ValueError('the init_type should be begin, end, and mid, but got {}'.format(self.init_type))
        for key in my_model_state_dict.keys():
            if key not in tea_state_dict:
                continue
            res = re.findall(pattern, key)
            if not res and not key.startswith('visual'):
                my_model_state_dict[key] = tea_state_dict[key]
            else:
                tea_key = re.sub(re.compile('\d'), map_layer(int(res[0])), string=key, count=1)
                my_model_state_dict[key] = tea_state_dict[tea_key]
        self.load_state_dict(my_model_state_dict)
        print('init with teacher weight success!')