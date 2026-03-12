# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
import json
import logging
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
import os
from fire import Fire
from infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from kvpress.ada_attn import replace_var_flash_attn
from kvpress.ada_cache import DynamicCacheSplitHeadFlatten
from loogle.calculate_metrics import calculate_metrics as loogle_scorer
from ruler.calculate_metrics import calculate_metrics as ruler_scorer
from tqdm import tqdm
from transformers import pipeline
from zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import time


from kvpress import (
    AdaKVPress,
    ChunkKVPress,
    CriticalKVPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    DuoAttentionPress,
    ExpectedAttentionPress,
    KnormPress,
    ObservedAttentionPress,
    RandomPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    EfficientDefensiveKVPress,
    EfficientLayerDefensiveKVPress,
    EfficientAdaSnapKVPress,
    EfficientAdaScorerPress,
    EfficientAdaGlobalScorerPress,
    ThinKPress,
    TOVAPress,
    CakeGlobalPress,
    # CakeScorerPress,
)

logger = logging.getLogger(__name__)

PRESS_DICT = {
    "criti_adasnapkv": CriticalAdaKVPress(SnapKVPress()),
    "criti_ada_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "criti_snapkv": CriticalKVPress(SnapKVPress()),
    "criti_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "adasnapkv": AdaKVPress(SnapKVPress()),
    "ada_expected_attention": AdaKVPress(ExpectedAttentionPress()),
    "expected_attention": ExpectedAttentionPress(),
    "ada_expected_attention_e2": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "random": RandomPress(),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "duo_attention": DuoAttentionPress(),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "efficient_ada_snapkv": EfficientAdaSnapKVPress(),
    "efficient_defensivekv": EfficientDefensiveKVPress(),
    "efficient_layer_defensivekv": EfficientLayerDefensiveKVPress(),
    "cake_global": CakeGlobalPress(),
    "fullkv": None,
}

@torch.inference_mode()
def efficiency_evaluate(
    model,
    tokenizer,
    press,
    context_length: int = 4 * 1024,
    compression_ratio: float = None,
    budget: int = None,
):
    """
    Evaluate a model on a dataset using a press and save the results
    """
    # assert compression_ratio is not None or budget is not None, "Either compression_ratio or budget must be provided"
    compression_ratio = 1 - budget / context_length if budget is not None else compression_ratio
    assert compression_ratio is None or compression_ratio > 0
    
    # fullkv press is None
    if press is not None:
        press.compression_ratio = compression_ratio

    prompt= "The quick brown fox jumps over the lazy dog." * (context_length)
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    input_ids = input_ids[:, :context_length]

    context_length=input_ids.shape[1]

   


    #NOTE cache budget = ( 1 - compression_ratio ) * context_length
    
    # Warmup: run several rounds before measurement
    warmup_rounds = 5
    for _ in range(warmup_rounds):
         # check if the press is an case of AdaKV
        if isinstance(press, EfficientAdaScorerPress):
            cache = DynamicCacheSplitHeadFlatten()
        elif isinstance(press, EfficientAdaGlobalScorerPress):
            cache = DynamicCacheSplitHeadFlatten()
        else:
            cache = DynamicCache()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        position_ids = torch.arange(
        0, context_length, device=model.device
        ).unsqueeze(0)
        with press(model) if press is not None else nullcontext():
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                position_ids=position_ids,
            )
        del outputs
        del cache
        del position_ids
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    
    # Measurement: run 10 rounds and take average
    measurement_rounds = 10
    prefill_latencies = []
    decoding_latencies = []
    max_memory_allocated_list = []
    max_memory_reserved_list = []
    
    for round_idx in range(measurement_rounds):
        # Reset cache for each round
        if isinstance(press, EfficientAdaScorerPress):
            cache = DynamicCacheSplitHeadFlatten()
        elif isinstance(press, EfficientAdaGlobalScorerPress):
            cache = DynamicCacheSplitHeadFlatten()
        else:
            cache = DynamicCache()
        
        # Reset position_ids for each round
        position_ids = torch.arange(
            0, context_length, device=model.device
        ).unsqueeze(0)
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t = time.time()
        # prefill and compress kv cache
        with press(model) if press is not None else nullcontext():
            outputs = model(
                input_ids=input_ids,
                past_key_values=cache,
                position_ids=position_ids,
                # num_logits_to_keep=1,
            )
        position_ids = position_ids[:, -1:] + 1
        generated_ids = [outputs.logits[0, -1].argmax()]
        # Clean up outputs to free memory
        del outputs
        torch.cuda.synchronize()
        t = time.time() - t
        prefill_latencies.append(t)

        torch.cuda.synchronize()
        t = time.time()
        ave_token_num = 100
        for i in range(100):
            outputs = model(
                input_ids=generated_ids[-1].unsqueeze(0).unsqueeze(0),
                past_key_values=cache,
                position_ids=position_ids + i,
            )
            # new_id = outputs.logits[0, -1].argmax()
            # Clean up outputs in each decoding step
            del outputs

        torch.cuda.synchronize()
        t = time.time() - t
        decoding_latencies.append(t / ave_token_num)
        max_memory_allocated_list.append(torch.cuda.max_memory_allocated("cuda") / (1024 ** 3))
        max_memory_reserved_list.append(torch.cuda.max_memory_reserved("cuda") / (1024 ** 3))
        
        # Clean up cache and other variables at the end of each round
        del cache
        del generated_ids
        del position_ids
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Calculate average
    prefill_latency = sum(prefill_latencies) / len(prefill_latencies)
    decoding_latency = sum(decoding_latencies) / len(decoding_latencies)
    max_memory_allocated = sum(max_memory_allocated_list) / len(max_memory_allocated_list)
    max_memory_reserved = sum(max_memory_reserved_list) / len(max_memory_reserved_list)
    return prefill_latency, decoding_latency, max_memory_allocated, max_memory_reserved



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate model efficiency with different press methods.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--press_name", type=str, required=True, choices=PRESS_DICT.keys(), help="Name of the press method to use.")
    args = parser.parse_args()

    model_path = args.model_path
    press_name = args.press_name
    device = 'cuda:0'

    if device is None: device = "cuda:0" if torch.cuda.is_available() else "cpu"


    press = PRESS_DICT[press_name]

    model_kwargs = {"attn_implementation": "flash_attention_2"}
    if isinstance(press, ObservedAttentionPress):
        model_kwargs = {"attn_implementation": "eager"}
    # Support AdaKV
    elif isinstance(press, EfficientAdaScorerPress):
        replace_var_flash_attn(model_name=model_path)
    elif isinstance(press, EfficientAdaGlobalScorerPress):
        replace_var_flash_attn(model_name=model_path)
    else:
        try:
            import flash_attn  # noqa: F401
            model_kwargs = {"attn_implementation": "flash_attention_2"}
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            **model_kwargs
            ).to("cuda:0")
    
    tokenizer=AutoTokenizer.from_pretrained(model_path)    

    print("press_name: ", press_name)
    budget_ratio = 0.5

    # 8*1024, 
    for context_length in [ 24*1024, 32*1024, 40*1024, ]:
        if press_name == "fullkv":
            budget = None
        else:
            budget = int(context_length * budget_ratio)

        try:
            prefill_latency, decoding_latency, max_memory_allocated, max_memory_reserved = efficiency_evaluate(model=model,tokenizer=tokenizer,press=press, budget = budget, context_length=context_length)

            print("=====================================")
            print("budget", budget, "context_length:", context_length, "prefill_latency(s)", f"{prefill_latency:.3f}", "decoding_latency(s)", f"{decoding_latency:.3f}", "max_memory_allocated(GB)", f"{max_memory_allocated:.2f}", "max_memory_reserved(GB)", f"{max_memory_reserved:.2f}")
            print("=====================================")
        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
            print("=====================================")
            print("budget", budget, "context_length:", context_length, " OOM Error!")
            print("=====================================")
            # Clean up GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            continue
