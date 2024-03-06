from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DataArguments:
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
    dataset: str = field(
        default=None,
        metadata={"help": "A json file (s2s) or text file with the dataset to train on"}
    )
    block_size: int = field(
        default=512,
        metadata={"help": "size of the blocks the text is split into for training"},
    )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    tokenizer: Optional[str] = field(
        default=None
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    max_instant_params: int = field(
        default=0,
        metadata={"help": "Maximum amount of paramters to optimize per step in millions"}
    )
    noresize: Optional[bool] = field(
        default=False,
        metadata={"help": "Never resize tokenizer embeddings"}
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
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    resume: bool = field(default=False, metadata={"help": 'Resume from previous checkpoint'})
    ddp_find_unused_parameters: bool = field(default=True, metadata={"help": 'set if trainer should try to find unused parameters'})
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
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
    group_by_length: bool = field(default=False,
                                  metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    storage_fp16: bool = field(default=False, metadata={"help": 'Store untrained layers in 16bit'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    max_checkpoints: int = field(default=0, metadata={"help": 'the maximum amount of checkpoints to save'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    primary_device: str = field(default="cuda:0", metadata={"help": 'The primary device to use'})
    secondary_device: str = field(default="cuda:0", metadata={"help": 'The secondary device to use'})
    train_non_linear_layers: str = field(default=False, metadata={"help": 'train non linear layers'})
    flush_allocator: bool = field(default=False, metadata={"help": 'flush torches allocator on eatch iteration'})
