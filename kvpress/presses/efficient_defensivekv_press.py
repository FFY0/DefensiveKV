# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half

from kvpress.presses.efficient_ada_scorer_press import EfficientAdaScorerPress
from kvpress.presses.efficient_ada_global_scorer_press import EfficientAdaGlobalScorerPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class EfficientDefensiveKVPress(EfficientAdaScorerPress):

    window_size: int = 32
    kernel_size: int = 5

    def __str__(self):
        return f"EfficientDefensiveKVPress={self.compression_ratio}_win={self.window_size}_kerl={self.kernel_size}"

    ## Following CriticalKV, we select tokens jointly using value norms and attention scores in a two-stage procedure.
    def vwl1norm(self, values, module, scores, window_bias, ave_attn_weights):
        bsz, num_key_value_heads, q_len, _ = values.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        Wo = module.o_proj.weight.transpose(0, 1)
        Wo = Wo.view(module.config.num_attention_heads, module.config.head_dim, module.config.hidden_size)
        V = repeat_kv(values, num_key_value_groups)

        # We use head-wise computation instead of direct matmul to reduce the memory usage of WoV.
        # Future kernel fusion optimization could eliminate this intermediate variables to enhance performance.
        head_WoV_norm_list = []
        for head in range(V.size(1)):
            head_WoV = V[:, head, :, ...].matmul(Wo[head, ...].unsqueeze(0))
            head_WoV_norm = torch.norm(head_WoV, p=1, dim=-1)
            head_WoV_norm_list.append(head_WoV_norm)

        # b_size, num_heads, q_len , k_len
        WoV_norm = torch.stack(head_WoV_norm_list, dim=1)
        WoV_norm = WoV_norm.view(bsz, num_key_value_heads, module.num_key_value_groups, q_len).mean(dim=2)

        projected_norm_normalization = WoV_norm / WoV_norm.sum(dim=-2, keepdim=True)
        scores = scores * projected_norm_normalization

        ## Specifically, we first select tokens until the cumulative attention mass reaches a 90% threshold (CriticalKV stage 1), and then jointly select additional tokens based on value norms (stage 2).
        threshold = 0.9 - window_bias
        normalized_scores = ave_attn_weights / ave_attn_weights.sum(dim=-1, keepdim=True)
        batch_size, num_heads = normalized_scores.shape[:2]
        score_mask = torch.zeros_like(normalized_scores, dtype=torch.bool)
        # calculate cumsum
        sorted_scores, sorted_indices = torch.sort(normalized_scores, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_scores, dim=-1)
        mask = cumsum >= threshold
        count = torch.argmax(mask.to(torch.int32), dim=-1)
        count.clamp_(max=int(q_len * (1 - self.compression_ratio) - self.window_size))
        for b in range(batch_size):
            for h in range(num_heads):
                k = count[b, h].item()
                indices = sorted_indices[b, h, :k]
                score_mask[b, h, indices] = True
        scores = torch.where(score_mask, scores.max().item(), scores)

        return scores

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

        window_bias = attn_weights[..., -window_size:].sum(dim=-1).mean().item()

        return attn_weights, window_bias

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        cache_metadata = kwargs.get("metadata", None)
        assert cache_metadata is not None, "cache_metadata is required for AdaSnapKVPress"

        # Current implementation only allows to compress once
        # check if first time compression
        head_lens = cache_metadata.head_lens
        assert all(
            x == head_lens[0] for x in head_lens
        ), "Not all elements in head_lens are the same, implying multiple compressions"

        # convert to (bsz, num_key_value_heads, q_len, head_dim) for easy score
        keys = keys.view(
            cache_metadata.bsz, cache_metadata.num_key_value_heads, cache_metadata.head_lens[0], keys.shape[-1]
        )
        values = values.view(
            cache_metadata.bsz, cache_metadata.num_key_value_heads, cache_metadata.head_lens[0], keys.shape[-1]
        )

        bsz, num_key_value_heads, q_len, head_dim = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert q_len > self.window_size, "Query length should be greater than the window size"

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights, window_bias = self.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )
        # Average per grioup (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = attn_weights.view(
            bsz, num_key_value_heads, num_key_value_groups, self.window_size, q_len - self.window_size
        )
        ave_attn_weights = scores.mean(dim=2).mean(dim=-2)
        scores_base = scores.sum(dim=-1, keepdim=True)
        scores = F.avg_pool1d(
            scores.view(bsz * num_key_value_heads * num_key_value_groups, self.window_size, q_len - self.window_size),
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            stride=1,
        ).view(bsz, num_key_value_heads, num_key_value_groups, self.window_size, q_len - self.window_size)
        scores = (scores / scores.sum(dim=-1, keepdim=True)) * scores_base

        ## Defensive Mechanism
        # the first max is for num_key_value_groups, the second max is for window_size
        max_scores = scores.max(dim=2).values.max(dim=-2).values
        scores = max_scores.clamp(min=max_scores.mean(dim=-1, keepdim=True))
        ## Defensive Mechanism End

        ## Borrowed from CriticalKV
        scores = self.vwl1norm(values[..., : -self.window_size, :], module, scores, window_bias, ave_attn_weights)

        # Add back the observation window. Use max score to make sure the window is not pruned.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        # Flatten scores
        flatten_scores = scores.view(bsz, num_key_value_heads * q_len)

        return flatten_scores
