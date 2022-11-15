import math
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.nn.functional as f
from torch import nn
from torch.nn.parameter import Parameter

from .output import ControlOutput, AttentionOutput, TransformerOutput, VisionTransformerOutput, \
    TransformerLayerOutput


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


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

    def forward(self, hidden_states, control_output: ControlOutput, attention_mask=None):
        """
        calculate the multihead attention
        :param control_output: Type: ControlOutput, For Control the Transformer output
        :param hidden_states: input hidden_states
        :param attention_mask: calculate the attention with attention_mask if not None
        :return: a Tuple next_hidden_states, attn_score_res, attn_prob_res, value_map
        """
        q, k, v = f.linear(hidden_states, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        query_layer = self.transpose_for_scores(q)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        value_map = None
        return_attention_scores = None
        return_attention_probs = None
        #  calculate the value_map
        if control_output.need_value_map:
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
        if control_output.need_attn_prob:
            return_attention_probs = attention_probs
        if control_output.need_attn_score:
            return_attention_scores = attention_scores
        return AttentionOutput(context_layer, return_attention_scores, return_attention_probs, value_map)


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

    def attention(self, x: torch.Tensor, control_output: ControlOutput):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, attention_mask=self.attn_mask, control_output=control_output)

    def forward(self, x: torch.Tensor, control_output: ControlOutput):
        """

        :param control_output: Type: ControlOutput, For Control the Transformer output
        :param x: hidden state input
        :return: a TransformerLayerOutput. (x, attn_output_scores, attn_output_prob, value_map)
        """
        attention_output: AttentionOutput = self.attention(self.ln_1(x), control_output)
        x = x + attention_output.attention_output
        x = x + self.mlp(self.ln_2(x))
        return TransformerLayerOutput(x, attention_output.attention_scores,
                                      attention_output.attention_probs, attention_output.value_map)


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,
                 need_layers: Optional[List],
                 attn_mask: torch.Tensor = None, drop_out: float = 0.1):
        super().__init__()
        self.width = width
        self.layers = layers

        self.need_layers = need_layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask, drop_out) for _ in range(layers)])

    def forward(self, x: torch.Tensor, control_output: ControlOutput):
        """
        calculate the Transformer layers
        :param control_output: Type: ControlOutput, For Control the Transformer output
        :param x: hidden state input
        :return: a TransformerOutput(hidden state, attention_scores, attention_probs, representations, value_map)
        """
        value_map = None
        attention_scores = []
        representations = []
        attention_probs = []

        for i, layer in enumerate(self.resblocks):
            layers_output: TransformerLayerOutput = layer(x, control_output)
            x = layers_output.hidden_representation
            if i not in self.need_layers:
                continue
            if control_output.need_rep:
                representations.append(layers_output.hidden_representation)
            if control_output.need_attn_score:
                attention_scores.append(layers_output.attention_scores)
            if control_output.need_attn_prob:
                attention_probs.append(layers_output.attention_probs)
            value_map = layers_output.value_map
        return TransformerOutput(x, attention_scores, attention_probs, representations, value_map)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 need_layers: List, drop_out: float = 0.1):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, drop_out=drop_out, need_layers=need_layers)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, control_output: ControlOutput):
        """
        calculate the vit layers
        :param control_output:
        :param x: hidden state input
        :return: a VisionTransformerOutput
        (hidden state, attention_scores, attention_probs, representations, value_map ,embedding)
        """
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        embeddings = x
        x = self.ln_pre(x)
        vision_output: VisionTransformerOutput = self.transformer(x, control_output)
        x = self.ln_post(vision_output.last_representation[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return VisionTransformerOutput(last_representation=x,
                                       attention_scores=vision_output.attention_scores,
                                       attention_probs=vision_output.attention_probs,
                                       representations=vision_output.representations,
                                       value_map=vision_output.value_map,
                                       embedding=embeddings)
