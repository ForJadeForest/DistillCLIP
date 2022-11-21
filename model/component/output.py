from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ControlOutput:
    need_emb: bool = False
    need_attn_score: bool = False,
    need_value_map: bool = False,
    need_attn_prob: bool = False,
    need_rep: bool = False,


@dataclass
class VisionTransformerOutput:
    last_representation: torch.Tensor = None
    last_layer_output: torch.Tensor = None
    attention_scores: List[torch.Tensor] = None
    attention_probs: List[torch.Tensor] = None
    representations: List[torch.Tensor] = None
    value_map: torch.Tensor = None,
    embedding: torch.Tensor = None,


@dataclass
class TextTransformerOutput:
    last_representation: torch.Tensor = None
    last_layer_output: torch.Tensor = None,
    attention_scores: List[torch.Tensor] = None
    attention_probs: List[torch.Tensor] = None
    representations: List[torch.Tensor] = None
    value_map: torch.Tensor = None,
    embedding: torch.Tensor = None,


@dataclass
class AttentionOutput:
    attention_output: torch.Tensor = None
    attention_scores: torch.Tensor = None,
    attention_probs: torch.Tensor = None,
    value_map: torch.Tensor = None


@dataclass
class TransformerOutput:
    last_layer_output: torch.Tensor = None
    attention_scores: List[torch.Tensor] = None
    attention_probs: List[torch.Tensor] = None
    representations: List[torch.Tensor] = None
    value_map: torch.Tensor = None


@dataclass
class TransformerLayerOutput:
    hidden_representation: torch.Tensor = None
    attention_scores: torch.Tensor = None
    attention_probs: torch.Tensor = None
    value_map: torch.Tensor = None

@dataclass
class CLIPOutput:
    visual_output: VisionTransformerOutput = None
    text_output: TextTransformerOutput = None
    i2t_logits: torch.Tensor = None
    t2i_logits: torch.Tensor = None
