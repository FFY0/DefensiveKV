#!/bin/bash

# Meta-Llama-3.1-8B-Instruct
model_name="Mistral-7B-Instruct-v0.3"
model=${MODELS_DIR}/${model_name}


# When evaluating the head-wise cache eviction method, please note that you must run the efficient version, which is supported by our optimized CUDA kernel, to achieve actual performance gains.
# The official kvpress repository provides a mask-based simulation of head-wise cache eviction; while simple, this approach does not deliver real efficiency improvements. 
# For example, to benchmark the speed of Ada-KV, you should run efficient_ada_snapkv rather than ada_snapkv. 
press_name=("efficient_layer_defensivekv" "efficient_defensivekv" "criti_snapkv"  "efficient_ada_snapkv"  "snapkv" "fullkv")
# press_name=("efficient_defensivekv" )


for press in "${press_name[@]}"; do
    python efficiency_evaluate.py --press_name $press --model_path $model
done