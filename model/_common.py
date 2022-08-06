import math
from collections import OrderedDict

import torch
import torch.nn.functional as f
from torch import nn
from torch.nn.parameter import Parameter


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class AttentionOutput(nn.Module):
    def __init__(self, hidden_size, drop_prob):
        super(AttentionOutput, self).__init__()
        self.out_linear = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.out_linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class MultiheadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, drop_prob):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads)
            )

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.in_proj_weight = Parameter(torch.empty((3 * hidden_size, hidden_size)))
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.in_proj_bias = Parameter(torch.empty(3 * hidden_size))

        self.dropout = nn.Dropout(drop_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, need_attn_score=False, need_value_map=False,
                need_attn_prob=False):
        q, k, v = f.linear(hidden_states, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        attn_prob_res = None
        attn_score_res = None
        value_map = None

        #  calculate the value_map
        if need_value_map:
            value_map = torch.matmul(value_layer, value_layer.transpose(-1, -2))
            value_map = value_map / math.sqrt(self.attention_head_size)
            value_map = nn.Softmax(dim=-1)(value_map)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = self.out_proj(context_layer)

        if need_attn_score:
            attn_score_res = attention_scores
        if need_attn_prob:
            attn_prob_res = attn_prob_res

        return context_layer, attn_score_res, attn_prob_res, value_map


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_prob: float = 0.1):
        super().__init__()

        self.attn = MultiheadAttention(d_model, n_head, drop_prob)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, need_attn_score=False, need_value_map=False, need_attn_prob=False):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, attention_mask=self.attn_mask, need_attn_score=need_attn_score,
                         need_value_map=need_value_map, need_attn_prob=need_attn_prob)

    def forward(self, x: torch.Tensor, need_attn_score=False, need_value_map=False, need_attn_prob=False):
        attn_output, attn_output_scores, attn_output_prob, value_map = self.attention(self.ln_1(x),
                                                                                      need_attn_score,
                                                                                      need_value_map,
                                                                                      need_attn_prob)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_output_scores, attn_output_prob, value_map


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, need_attn_score=False, need_value_map=False, need_attn_prob=False,
                need_rep=False):
        attention_scores = []
        representations = []
        attention_probs = []
        value_map = None
        for layer in self.resblocks:
            x, attn, attn_prob, value_map = layer(x, need_attn_score, need_attn_prob, need_value_map)
            if need_rep:
                representations.append(x)
            if need_attn_score:
                attention_scores.append(attn)
            if need_attn_prob:
                attention_probs.append(attn_prob)
        return x, attention_scores, attention_probs, representations, value_map


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, need_attn_score=False, need_value_map=False, need_attn_prob=False,
                need_rep=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        embeddings = x
        x = self.ln_pre(x)
        x, attention_maps, attention_probs, representations, value_maps = self.transformer(x, need_attn_score,
                                                                                           need_value_map,
                                                                                           need_attn_prob,
                                                                                           need_rep)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x, attention_maps, representations, embeddings, attention_probs, value_maps


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
