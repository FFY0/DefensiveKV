  
# export KVPRESS_DATASETS=defensive_kvpress_release/defensivekv_dataset
# export MODELS_DIR=defensive_kvpress_release/defensivekv_models

dataset="ruler"
data_dir=${KVPRESS_DATASETS}/ruler/4096/
compression_ratios=(0.8)
model_names=( "Meta-Llama-3.1-8B-Instruct" )
press_names_group1=("efficient_layer_defensivekv" "efficient_defensivekv" "criti_adasnapkv"  "adasnapkv" "snapkv")
fraction=0.1
LOG_DIR="logs"

for compression_ratio in "${compression_ratios[@]}"; do
    for model_name in "${model_names[@]}"; do
      model_path="${MODELS_DIR}/${model_name}"
      echo model_path: $model_path
      for i in "${!press_names_group1[@]}"; do
        press="${press_names_group1[$i]}"
        echo "Running $model_name press_name: $press with compression_ratio: $compression_ratio "
        outname="${dataset}_${model_name}_${press}_${compression_ratio}_fraction${fraction}"
        CUDA_VISIBLE_DEVICES=0 python evaluate.py --dataset $dataset --data_dir $data_dir --model $model_path --press_name $press --compression_ratio $compression_ratio --device "cuda:0" --fraction $fraction > $LOG_DIR/$outname.log 2>&1 
      done
    done
  done



