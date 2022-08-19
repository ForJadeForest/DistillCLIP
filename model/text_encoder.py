import torch
from torch import nn

try:
    from _common import Transformer, LayerNorm
    from _utils import output_filter
except ModuleNotFoundError:
    from ._common import Transformer, LayerNorm
    from ._utils import output_filter


class TextEncoder(nn.Module):
    def __init__(self, transformer_width, transformer_layers, transformer_heads, context_length, vocab_size, embed_dim,
                 tea_transformer_width=None, is_student=True, drop_out=0.1):
        super().__init__()
        self.context_length = context_length
        self.transformer_width = transformer_width
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            drop_out=drop_out
        )
        self.layers = transformer_layers
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

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def encode_text(self, text, only_last_state=True, need_attn_score=False, need_value_map=False, need_attn_prob=False,
                    need_rep=False, need_emb=False):
        embedding = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = embedding + self.positional_embedding
        x, attention_maps, attention_probs, representations, value_map = self.transformer(x, need_attn_score,
                                                                                          need_value_map,
                                                                                          need_attn_prob,
                                                                                          need_rep)
        x = self.ln_final(x)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        if only_last_state:
            return x

        return output_filter(self.is_student, representations, self.embedding_projection, embedding, attention_maps, x,
                             self.hidden_projection, attention_probs, value_map, need_emb, need_rep, need_attn_score,
                             self.no_trans)

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

    def forward(self, text, only_last_state=True, **need_para):
        return self.encode_text(text, only_last_state, **need_para)

    def hyper_para(self):
        return {
            'context_length': self.context_length,
            'transformer_width': self.transformer_width,
            'transformer_layers': self.transformer_layers,
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


if __name__ == '__main__':
    m = TextEncoder(512, 6, 8, 77, 49408, 512, 768, True)
    # inputs = torch.randint(low=0, high=300, size=(64, 77))
    # outputs = m(inputs, False)
    # print(len(outputs), outputs[0].shape)
    from _utils import LayerMap
    layer_map = LayerMap(6, 12, 'mid')
    from _utils import load
    state_dict = load('ViT-B/32', download_root='D:/data/cache')
    para = {
        'embed_dim': state_dict["text_projection"].shape[1],
        'context_length': state_dict["positional_embedding"].shape[0],
        'vocab_size': state_dict["token_embedding.weight"].shape[0],
        'transformer_width': state_dict["ln_final.weight"].shape[0],
        'transformer_heads': state_dict["ln_final.weight"].shape[0] // 64,
        'transformer_layers': len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks"))),
    }
    teacher_model = TextEncoder(is_student=False, **para)
    my_state_dict = teacher_model.state_dict()
    for k in my_state_dict:
        if k in state_dict:
            my_state_dict[k] = state_dict[k]
    teacher_model.load_state_dict(my_state_dict)

    m.init_layers_with_teacher(layer_map, teacher_model.state_dict(), 'begin')
    m.init_layers_with_teacher(layer_map, teacher_model.state_dict(), 'mid')
    m.init_layers_with_teacher(layer_map, teacher_model.state_dict(), 'end')

