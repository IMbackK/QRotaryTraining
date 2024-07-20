
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

from dataclasses import dataclass, field
from typing import Optional, Self
from enum import Enum


class DatasetType(Enum):
    TEXT = 1
    S2S = 2
    HUB = 3
    CHAT = 4

    @staticmethod
    def to_string(dtype: Self) -> str:
        if dtype == DatasetType.TEXT:
            return "text"
        elif dtype == DatasetType.S2S:
            return "s2s"
        elif dtype == DatasetType.HUB:
            return "hub"
        elif dtype == DatasetType.CHAT:
            return "chat"
        return "invalid"

    @staticmethod
    def from_string(string: str):
        if string == str(DatasetType.TEXT):
            return DatasetType.TEXT
        elif string == str(DatasetType.S2S):
            return DatasetType.S2S
        elif string == str(DatasetType.HUB):
            return DatasetType.HUB
        elif string == str(DatasetType.CHAT):
            return DatasetType.CHAT
        return None

    def __str__(self):
        return DatasetType.to_string(self)


@dataclass
class DataArguments:
    dataset: str = field(
        metadata={"help": "The dataset to train on"}
    )
    dataset_type: str = field(
        default="text", metadata={"help": f"The type of dataset, set to one of {[e for e in DatasetType]}"}
    )
    dataset_chat_template: str | None = field(
        default=None, metadata={"help": "overrides the chat template to be the one set here"}
    )
    eval_dataset_size: int = field(
        default=512, metadata={"help": "Size of validation dataset."}
    )
    source_max_len: int = field(
        default=512,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Wether to train on the input in addition to the target text when in s2s mode."}
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    data_from_hub: Optional[bool] = field(
        default=False,
        metadata={"help": "If this is set the dataset is assumed to be a name of a hf-hub dataset"}
    )


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="EleutherAI/pythia-12b"
    )
    tokenizer: Optional[str] = field(
        default=None
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    noresize: Optional[bool] = field(
        default=False,
        metadata={"help": "Never resize tokenizer embeddings"}
    )
    quantize: bool = field(
        default=False,
        metadata={"help": "Quantize parameters not currently be actively trained"}
    )


@dataclass
class TrainingArguments():
    cache_dir: Optional[str] = field(
        default=None
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    resume: bool = field(default=False, metadata={"help": 'Resume from previous checkpoint'})
    ddp_find_unused_parameters: bool = field(default=True, metadata={"help": 'set if trainer should try to find unused parameters'})
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for checkpoints'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    per_device_eval_batch_size: int = field(default=1, metadata={"help": 'The eval batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    epochs: int = field(default=3, metadata={"help": 'How many epochs to train for'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    adam_epsilon: float = field(default=1e-7, metadata={"help": 'Adam epsilon'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    fp16: bool = field(default=False, metadata={"help": 'Train in 16 bit mixed precision'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    do_eval: bool = field(default=False, metadata={"help": 'To eval or not to eval, that is the question?'})
    lr_scheduler_type: str = field(default='constant',
                                   metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_steps: float = field(default=0, metadata={"help": 'number of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    logging_dir: str = field(default='./log', metadata={"help": 'The output dir for logs'})
    group_by_length: bool = field(default=False,
                                  metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    max_checkpoints: int = field(default=0, metadata={"help": 'the maximum amount of checkpoints to save'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    train_non_linear_layers: Optional[bool] = field(default=False, metadata={"help": 'train non linear layers'})
    flush_allocator: bool = field(default=False, metadata={"help": 'flush torches allocator on eatch iteration'})
    max_instant_params: int = field(default=0, metadata={"help": "Maximum amount of paramters to optimize per step in millions"})
    churn_percent: int = field(default=100, metadata={"help": "The percentage of active parameters to replace when changeing active parameters"})
    eval_steps: int = field(default=-1, metadata={"help": "Number of optimization steps after wich to compute the evaluation loss"})
    eval_prompt: str | None = field(default=None, metadata={"help": "A prompt to used during eval to check if the model is learning"})
    reshufle_steps: int = field(default=50, metadata={"help": "Number of steps to take before changing the active parameters"})
