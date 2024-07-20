#!/bin/sh

BASE_DIR=$(dirname "$0")
VENV_DIR=$(venvget)

export MAX_JOBS=48

export ROCR_VISIBLE_DEVICES="1,2"
source $VENV_DIR/bin/activate

python $SCRIPTS/train_dyamic/train_dynamic.py \
	--model_name_or_path "huggyllama/llama-7b" \
	--dataset "tatsu-lab/alpaca" \
	--data_from_hub \
	--do_train \
	--reshufle_steps 50 \
	--per_device_train_batch_size 8 \
	--per_device_eval_batch_size 8 \
	--gradient_checkpointing False \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 2 \
	--logging_dir $BASE_DIR/log \
	--logging_steps 5 \
	--learning_rate 1e-6 \
	--save_steps 1000 \
	--output_dir $BASE_DIR/llama-7b \
	--adam8bit \
	--max_instant_params 2000\
	--churn_percent 50\
	--quantize
