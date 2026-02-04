# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from kvpress.attention_patch import patch_attention_functions
from kvpress.pipeline import KVPressTextGenerationPipeline
from kvpress.presses.adakv_press import AdaKVPress
from kvpress.presses.base_press import BasePress
from kvpress.presses.chunk_press import ChunkPress
from kvpress.presses.chunkkv_press import ChunkKVPress
from kvpress.presses.composed_press import ComposedPress
from kvpress.presses.criticalkv_press import CriticalAdaKVPress, CriticalKVPress
from kvpress.presses.duo_attention_press import DuoAttentionPress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.key_rerotation_press import KeyRerotationPress
from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.observed_attention_press import ObservedAttentionPress
from kvpress.presses.per_layer_compression_press import PerLayerCompressionPress
from kvpress.presses.random_press import RandomPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.simlayerkv_press import SimLayerKVPress
from kvpress.presses.snapkv_press import SnapKVPress
from kvpress.presses.streaming_llm_press import StreamingLLMPress
from kvpress.presses.think_press import ThinKPress
from kvpress.presses.composed_press import ComposedPress
from kvpress.presses.tova_press import TOVAPress
from kvpress.presses.efficient_ada_scorer_press import EfficientAdaScorerPress
from kvpress.presses.efficient_ada_global_scorer_press import EfficientAdaGlobalScorerPress
from kvpress.presses.efficient_ada_snapkv_press import EfficientAdaSnapKVPress
from kvpress.presses.efficient_defensivekv_press import EfficientDefensiveKVPress
from kvpress.presses.efficient_layer_defensivekv_press import EfficientLayerDefensiveKVPress
from kvpress.presses.cake_scorer_press import CakeScorerPress
from kvpress.presses.cake_global_press import CakeGlobalPress

from kvpress.presses.tova_press import TOVAPress
from kvpress.presses.qfilter_press import QFilterPress


# Patch the attention functions to support head-wise compression
patch_attention_functions()

__all__ = [
    "BasePress",
    "CriticalKVPress",
    "ComposedPress",
    "ScorerPress",
    "KnormPress",
    "ObservedAttentionPress",
    "RandomPress",
    "SimLayerKVPress",
    "SnapKVPress",
    "StreamingLLMPress",
    "ThinKPress",
    "TOVAPress",
    "KVPressTextGenerationPipeline",
    "PerLayerCompressionPress",
    "KeyRerotationPress",
    "DuoAttentionPress",
    "QFilterPress",
    # Easy Implementation for AdaKV
    "AdaKVPress",
    "CriticalAdaKVPress",
    # Efficient Implementation for Head-wise Method, AdaKV variants
    "EfficientDefensiveKVPress",
    "EfficientLayerDefensiveKVPress",
    "EfficientAdaScorerPress",
    "EfficientAdaGlobalScorerPress",
    "EfficientAdaSnapKVPress",
    "ExpectedAttentionPress",
    "CakeGlobalPress",
    "CakeScorerPress",

]
