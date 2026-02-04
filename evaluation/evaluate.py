# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from pathlib import Path
import time
from typing import Optional

import torch
from datasets import load_dataset, load_from_disk
import pandas as pd
from fire import Fire
import transformers
from infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from longbench.calculate_metrics import calculate_metrics as longbench_scorer
from kvpress.ada_attn import replace_var_flash_attn
from loogle.calculate_metrics import calculate_metrics as loogle_scorer
from ruler.calculate_metrics import calculate_metrics as ruler_scorer
from tqdm import tqdm
from transformers import pipeline
from zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer
import warnings

warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

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

DATASET_DICT = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": None,
}

SCORER_DICT = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
}

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
}



def evaluate(
    dataset: str = "ruler",
    # dataset: str = "longbench",
    data_dir: Optional[str] = "/datasets/SlowMov/ruler/4096/",
    model: str = "/models/Meta-Llama-3.1-8B-Instruct",
    device: Optional[str] = None,
    press_name: str = "efficient_ada_denfensive",
    compression_ratio: float = 0.75,
    fraction: float = 0.2,
    max_new_tokens: Optional[int] = None,
    max_context_length: Optional[int] = None,
    compress_questions: bool = False,
):
    """
    Evaluate a model on a dataset using a press and save the results

    Parameters
    ----------
    dataset : str
        Dataset to evaluate
    data_dir : str, optional
        Subdirectory of the dataset to evaluate, by default None
    model : str, optional
        Model to use, by default "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device : str, optional
        Model device, by default cuda:0 if available else cpu. For multi-GPU use "auto"
    press_name : str, optional
        Press to use (see PRESS_DICT), by default "expected_attention"
    compression_ratio : float, optional
        Compression ratio for the press, by default 0.1
    max_new_tokens : int, optional
        Maximum number of new tokens to generate, by default use the default for the task (recommended)
    fraction : float, optional
        Fraction of the dataset to evaluate, by default 1.0
    max_context_length : int, optional
        Maximum number of tokens to use in the context. By default will use the maximum length supported by the model.
    compress_questions : bool, optional
        Whether to compress the questions as well, by default False
    """
    assert dataset in DATASET_DICT, f"No dataset found for {dataset}"
    assert dataset in SCORER_DICT, f"No scorer found for {dataset}"
    data_dir = str(data_dir) if data_dir else None
    # Load press
    if press_name is not None:
        assert press_name in PRESS_DICT
        press = PRESS_DICT[press_name]
        if isinstance(press, (DuoAttentionPress)):
            press.head_compression_ratio = compression_ratio
        else:
            press.compression_ratio = compression_ratio  # type:ignore[attr-definedif press is not None
    else:
        press = None

    if device is None:
        device = "cuda:7" if torch.cuda.is_available() else "cpu"

    save_dir = Path(__file__).parent / "results"
    save_dir.mkdir(exist_ok=True)
    save_filename = save_dir / (
        "__".join(
            [
                dataset,
                model.replace("/", "--").split("--")[-1],
                str(press),
                str(compression_ratio),
                f"frac{fraction:.2f}",
            ]
        )
        + ".csv"
    )
    print("try save to:", save_filename)
    if save_filename.exists():
        logger.warning(f"Results already exist at {save_filename}")
        print("Results already exist at", save_filename)
        exit()

    # Load dataframe
    try:
        print("Loading from disk, data_dir:", data_dir)
        df = load_from_disk(data_dir).to_pandas()
    except Exception as e:
        print(f"Failed to load from disk: {e}")
        exit()

    if fraction < 1.0:
        # Stratified sampling by task category
        sampled_dfs = []
        for task_name, task_df in df.groupby("task"):
            sampled_task_df = task_df.sample(frac=fraction, random_state=42)
            sampled_dfs.append(sampled_task_df)
        df = pd.concat(sampled_dfs)
        save_filename = save_filename.with_name(save_filename.stem + f"__fraction{fraction:.2f}" + save_filename.suffix)

    if max_context_length is not None:
        save_filename = save_filename.with_name(
            save_filename.stem + f"__max_context{max_context_length}" + save_filename.suffix
        )

    if compress_questions:
        df["context"] = df["context"] + df["question"]
        df["question"] = ""
        save_filename = save_filename.with_name(save_filename.stem + "__compressed_questions" + save_filename.suffix)

    # Initialize pipeline with the correct attention implementation
    model_kwargs = {}
    if isinstance(press, ObservedAttentionPress):
        model_kwargs = {"attn_implementation": "eager"}
    # Support AdaKV
    elif isinstance(press, EfficientAdaScorerPress):
        replace_var_flash_attn(model_name=model)
    elif isinstance(press, EfficientAdaGlobalScorerPress):
        replace_var_flash_attn(model_name=model)
    else:
        try:
            import flash_attn  # noqa: F401

            model_kwargs = {"attn_implementation": "flash_attention_2"}
        except ImportError:
            pass

    model_kwargs["torch_dtype"] = "auto"
    if device == "auto":
        pipe = pipeline("kv-press-text-generation", model=model, device_map="auto", model_kwargs=model_kwargs)
    else:
        pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

    print("model dtype: ", pipe.model.dtype, flush=True)
    # Run pipeline on each context
    df["predicted_answer"] = None
    df_context = df.groupby("context")
    assert all(df_context["answer_prefix"].nunique() == 1)

    if dataset == "longbench": 
        # evalutated_tasks = ["qasper"]
        evalutated_tasks = None # Test all
    elif dataset == "ruler":
        # evalutated_tasks = ["niah_multivalue"]
        evalutated_tasks = None # Test all
    else:
        evalutated_tasks = None
            
    for context, df_ in tqdm(df_context, total=df["context"].nunique()):

        task_name = df_["task"].iloc[0]

        # skip specific tasks, which are not in the task_names
        if evalutated_tasks is not None:
            if task_name not in evalutated_tasks:
                continue

        chat_template_bak = pipe.tokenizer.chat_template
        bos_bak = pipe.tokenizer.bos_token
        gen_config_eos_id_bak = pipe.model.generation_config.eos_token_id

        if task_name in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            pipe.tokenizer.chat_template = None
            pipe.tokenizer.bos_token = ""
            if task_name in ["samsum"]:
                pipe.model.generation_config.eos_token_id = [
                    pipe.tokenizer.eos_token_id,
                    pipe.tokenizer.encode("\n", add_special_tokens=False)[-1],
                ]

        questions = df_["question"].to_list()
        max_new_tokens_ = max_new_tokens if max_new_tokens is not None else df_["max_new_tokens"].iloc[0]
        answer_prefix = df_["answer_prefix"].iloc[0]
        Failure_count = 0
        try:
            output = pipe(
                context,
                questions=questions,
                answer_prefix=answer_prefix,
                press=press,
                max_new_tokens=max_new_tokens_,
                max_context_length=max_context_length,
            )
        except Exception as e:
            print("An error occurred:", e)
            output = {"answers": "Failure:" + str(e)}
            Failure_count += 1

        df.loc[df_.index, "predicted_answer"] = output["answers"]
        if press:
            df.loc[df_.index, "compression_ratio"] = press.compression_ratio  # type:ignore[attr-defined]
        else:
            df.loc[df_.index, "compression_ratio"] = 0  # type:ignore[attr-defined]

        # restore chat template
        pipe.tokenizer.chat_template = chat_template_bak
        pipe.tokenizer.bos_token = bos_bak
        pipe.model.generation_config.eos_token_id = gen_config_eos_id_bak

    # Save answers
    df[["predicted_answer", "compression_ratio"]].to_csv(str(save_filename), index=False)

    print("Saving DataFrame to", save_filename)

    df.to_csv(str(save_filename).replace(".csv", "_df.csv"), index=False)
    # Calculate metrics
    scorer = SCORER_DICT[dataset]
    metrics = scorer(df)
    with open(str(save_filename).replace(".csv", ".json"), "w") as f:
        json.dump(metrics, f)
    print(f"Average compression ratio: {df['compression_ratio'].mean():.2f}")
    print(f"Failure count: {Failure_count}")
    print(metrics)


if __name__ == "__main__":
    Fire(evaluate)
