# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass
import os

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.cake_scorer_press import CakeScorerPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class CakeGlobalPress(CakeScorerPress):

    window_size: int = 32
    kernel_size: int = 5
    # n_sink: int = 4
    tau1: float = 1.6
    tau2: float = 0.6
    gamma: float = 200.0

    def __str__(self):
        return f"CakeGlobalPress_com_ratio={self.compression_ratio}, wind_size={self.window_size}, kerl_size={self.kernel_size}"

    @staticmethod
    def calculate_entropy(attention_scores):
        attention_scores = attention_scores.to(torch.float32)
        entropy = -torch.sum(attention_scores * torch.log(attention_scores + 1e-10))
        entropy = entropy.to(dtype=torch.float32)
        return entropy

    @staticmethod
    def compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
        """
        Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
        """

        bsz, q_len, _ = hidden_states.shape
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        num_key_value_groups = num_heads // module.config.num_key_value_heads

        # Get last window_size queries
        if hasattr(module, "q_proj"):
            query_states = module.q_proj(hidden_states[:, -window_size:])
        elif hasattr(module, "qkv_proj"):
            qkv = module.qkv_proj(hidden_states[:, -window_size:])
            query_states = qkv[..., : num_heads * head_dim]
        else:
            raise NotImplementedError(f"SnapKV not yet implemented for {module.__class__}.")

        query_states = query_states.view(bsz, window_size, num_heads, head_dim).transpose(1, 2)

        # Apply RoPE
        cos, sin = position_embeddings
        cos, sin = cos[:, -window_size:], sin[:, -window_size:]
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

        # Compute attention for first q_len - window_size tokens
        key_states = repeat_kv(keys, num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
        attention_mask = torch.ones_like(attn_weights) * float("-inf")
        attention_mask = torch.triu(attention_mask, diagonal=q_len - window_size + 1)
        attn_weights += attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = attn_weights[..., :-window_size]

        return attn_weights

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        # Current implementation only allows to compress once
        # check if first time compression
        bsz, q_len, _ = hidden_states.shape

        # convert to (bsz, num_key_value_heads, q_len, head_dim) for easy score
        keys = keys.view(bsz, module.config.num_key_value_heads, q_len, keys.shape[-1])
        values = values.view(bsz, module.config.num_key_value_heads, q_len, keys.shape[-1])

        bsz, num_key_value_heads, q_len, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert q_len > self.window_size, "Query length should be greater than the window size"

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        disp = self.calculate_entropy(attn_weights[:, :, -self.window_size :, :])
        var = torch.var(attn_weights[:, :, -self.window_size :, :], dim=-2).sum(0).sum(0).sum(0)
        pref_score = (disp ** (1 / self.tau1) * var ** (1 / self.tau2)).cpu().numpy()

        attn_scores = attn_weights[:, :, -self.window_size :, :]
        attn_mean = attn_scores.mean(dim=-2)
        attn_var = attn_scores.var(dim=-2)
        attn_cache = attn_mean + self.gamma * attn_var
        # attn_cache = attn_cache[:, :, :-self.window_size]
        attn_cache = F.avg_pool1d(attn_cache, kernel_size=5, padding=5 // 2, stride=1)
        attn_cache = attn_cache.reshape(bsz, num_key_value_heads, num_key_value_groups, -1)
        hh_score = attn_cache.mean(dim=-2)

        return hh_score, pref_score
