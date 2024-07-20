#!/bin/sh

BASE_DIR=$(dirname "$0")
VENV_DIR=$(venvget)

export MAX_JOBS=48

export ROCR_VISIBLE_DEVICES="1,2"
source $VENV_DIR/bin/activate

python $SCRIPTS/train_dyamic/train_dynamic.py \
	--model_name_or_path "huggyllama/llama-7b" \
	--dataset "tatsu-lab/alpaca" \
	--dataset_type "hub" \
	--eval_dataset_size 200 \
	--source_max_len 1024 \
	--do_train \
	--do_eval \
	--eval_steps 100 \
	--reshufle_steps 50 \
	--per_device_train_batch_size 2 \
	--per_device_eval_batch_size 1 \
	--gradient_checkpointing True \
	--gradient_accumulation_steps 4 \
	--epochs 3 \
	--logging_dir $BASE_DIR/log \
	--logging_steps 5 \
	--learning_rate 1e-6 \
	--save_steps 500 \
	--output_dir $BASE_DIR/llama-7b-quant \
	--adam8bit \
	--churn_percent 100\
	--max_instant_params 3000 \
	--quantize
