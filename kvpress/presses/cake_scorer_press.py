from functools import cache
import logging
from dataclasses import dataclass
import os

import torch
from torch import nn
import numpy as np

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class CakeScorerPress(BasePress):

    compression_ratio: float = 0.0
    window_size: int = 32
    hh_scores = []
    pref_scores= []

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, "Compression ratio must be between 0 and 1"

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute a tensor of fallened scores with shape (bsz, num_key_value_heads * q_len).
        The KV pairs with lowest scores **among all heads in one layer** will be adaptively pruned in the `compress` method.
        """
        raise NotImplementedError
    
    @staticmethod
    def adjust_budgets(budget_list, total_budget, seq_len, layer_nums):

        budget_list = np.array(budget_list, dtype=int)
        # Limit the budget of all layers to not exceed seq_len
        excess = np.maximum(budget_list - seq_len, 0)
        budget_list = np.minimum(budget_list, seq_len)

        # Adjust excess budget
        total_excess = np.sum(excess)

        if total_excess > 0:

            valid_indices = budget_list < seq_len
            num_valid = np.sum(valid_indices)

            if num_valid > 0:
                
                distribute_per_layer = total_excess // num_valid
                remainder = total_excess % num_valid

                budget_list[valid_indices] += distribute_per_layer
                budget_list[np.where(valid_indices)[0][:remainder]] += 1

        # Ensure total budget equals total_budget
        current_total_budget = np.sum(budget_list)
        budget_diff = total_budget - current_total_budget

        if budget_diff != 0:
            if budget_diff > 0:
                valid_indices = budget_list < seq_len  
            else:
                valid_indices = budget_list > 1  

            num_valid = np.sum(valid_indices)

            if num_valid > 0:
                adjust_per_layer = abs(budget_diff) // num_valid
                remainder = abs(budget_diff) % num_valid

                if budget_diff > 0:
                    budget_list[valid_indices] += adjust_per_layer
                    budget_list[np.where(valid_indices)[0][:remainder]] += 1
                else:
                    budget_list[valid_indices] -= adjust_per_layer
                    budget_list[np.where(valid_indices)[0][:remainder]] -= 1

        return budget_list.tolist()
    
    
    def layer_compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        # attentions: torch.Tensor,
        budget: int,
        layer_idx: int, 
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        cache = kwargs.get("past_key_value", None)
        
        keys = cache.key_cache[layer_idx]
        values = cache.value_cache[layer_idx]
        
        if self.compression_ratio == 0:
            return keys, values
        
        bsz, num_key_value_heads, seq_len, head_dim = cache.key_cache[layer_idx].shape
        
        hh_score = self.hh_scores[layer_idx]

        if budget > hh_score.shape[-1]:
            budget = hh_score.shape[-1]
  
        indices = hh_score.topk(budget, dim=-1).indices

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        k_past_compress = cache.key_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = cache.value_cache[layer_idx][:, :, :-self.window_size, :].gather(dim=2, index=indices)
        k_cur = cache.key_cache[layer_idx][:, :, -self.window_size:, :]
        v_cur = cache.value_cache[layer_idx][:, :, -self.window_size:, :]
        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)
        
        # update origin cache
        cache.key_cache[layer_idx] = key_states
        cache.value_cache[layer_idx] = value_states

        return key_states, value_states
        

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        The `compress` function adaptively compresses the cache based on scores following the Ada-KV Paradigm.
        It selects the top-k keys and values among all heads in a layer based on the scores, achieving head-specific compression.

        Example:
            - Batch size (bsz) = 1
            - Number of key-value heads = 2
            - Sequence length (seqlen) = 4
            - Cache budget = 4

        Given:
            (cache) scores = [[head1: [3, 4, 5, 9999], head2: [1, 1, 1, 9998]]]

        The compression process results in:
            compressed (cache) scores = [[head1: [4, 5, 9999], head2: [9998]]]
            flattened (cache) scores = [[4, 5, 9999, 9998]]
        """

        if self.compression_ratio == 0:
            return keys, values

        cache = kwargs.get("past_key_value", None)

        with torch.no_grad():
            # kwargs["metadata"] = cache_metadata
            hh_score, pref_score = self.score(module, hidden_states, keys, values, attentions, kwargs)
                        
        # only perform compression in the last layer    
        self.hh_scores.append(hh_score)
        self.pref_scores.append(pref_score)
        if module.layer_idx < module.config.num_hidden_layers - 1:
            return keys, values
        
        # q_len = hidden_states.shape[1]
        bsz, num_key_value_heads, q_len, head_dim = keys.shape
        
        # Calculate overall budget for total layers
        n_kept = int((q_len * (1 - self.compression_ratio) - self.window_size)* module.config.num_hidden_layers) 
        # NOTE: current implementation only support bsz 1
        # assert flatten_scores.shape[0] == 1
        
        layer_budgets = [pref_score/sum(self.pref_scores)*n_kept for pref_score in self.pref_scores]
        layer_budgets = self.adjust_budgets(layer_budgets, n_kept, q_len-self.window_size, module.config.num_hidden_layers)
    
        # intra-layer Budget allocation
        for layer_idx in range(module.config.num_hidden_layers):
            budget = int(layer_budgets[layer_idx])
            if budget >= q_len - self.window_size:
                budget = q_len - self.window_size
            self.layer_compress(module, hidden_states, keys, values, budget, layer_idx, kwargs)
        
        self.hh_scores = []
        self.pref_scores= []
        keys = cache.key_cache[module.layer_idx]
        values = cache.value_cache[module.layer_idx]
        
        return keys, values