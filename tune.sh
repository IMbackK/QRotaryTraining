#!/bin/sh
#
# QRotaryTraining - A novel method for fully training all parameters of large
# language models (llms) while using less device memory than traditional methods.
# Copyright (C) 2024 Carl Philipp Klemm
#
# This file is part of QRotaryTraining.
#
# QRotaryTraining is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# QRotaryTraining is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with QRotaryTraining.  If not, see <http://www.gnu.org/licenses/>.
#

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
