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
    attention_scores: List[torch.Tensor] = [],
    attention_probs: List[torch.Tensor] = [],
    representations: List[torch.Tensor] = [],
    value_map: torch.Tensor = None,
    embedding: torch.Tensor = None,


@dataclass
class TextTransformerOutput:
    last_representation: torch.Tensor = None
    attention_scores: List[torch.Tensor] = [],
    attention_probs: List[torch.Tensor] = [],
    representations: List[torch.Tensor] = [],
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
    last_representation: torch.Tensor = None
    attention_scores: List[torch.Tensor] = [],
    attention_probs: List[torch.Tensor] = [],
    representations: List[torch.Tensor] = [],
    value_map: torch.Tensor = None


@dataclass
class TransformerLayerOutput:
    hidden_representation: torch.Tensor = None
    attention_scores: torch.Tensor = None,
    attention_probs: torch.Tensor = None,
    value_map: torch.Tensor = None
