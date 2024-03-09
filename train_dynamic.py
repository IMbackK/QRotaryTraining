import transformers
from transformers import AutoModelForCausalLM, get_scheduler
from peft.utils import _get_submodules

import torch
from torch.utils import tensorboard
import os
import shutil
import math
import collections
from tqdm.auto import tqdm
from random import randint
import collections


from arguments import DataArguments, ModelArguments, TrainingArguments
from datamodules import create_data_module_s2s, create_data_module
from convertinglinear import ConvertingLinear
from tokenizer import get_tokenizer


def find_all_linear_module_names(model):
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, ConvertingLinear):
            module_names.add(name)

    if 'lm_head' in module_names:  # needed for 16-bit
        module_names.remove('lm_head')
    return list(module_names)


def find_all_outher_module_names(model):
    module_names = set()
    for name, module in model.named_modules():
        if not (isinstance(module, torch.nn.Linear) or isinstance(module, ConvertingLinear)):
            module_names.add(name)
    return list(module_names)


def get_model(model_args: ModelArguments, cache_dir, gradient_checkpointing):
    dtype = torch.float16 if training_args.fp16 or (training_args.storage_fp16 and model_args.max_instant_params > 0) else torch.float32
    print(f'loading base model {model_args.model_name_or_path} in {dtype}...')

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=cache_dir,
        torch_dtype=dtype if model_args.max_instant_params > 0 else torch.float32,
        trust_remote_code=model_args.trust_remote_code,
        device_map=None,
        attn_implementation="flash_attention_2"
    )

    modules = dict(model.named_modules())
    keys = find_all_linear_module_names(model)
    for key in keys:
        modules[key].weight = modules[key].weight.to(dtype)
        if modules[key].bias is not None:
            modules[key].bias = modules[key].bias.to(dtype)

    return model


@torch.no_grad()
def recursive_setattr(obj, attr, value):
    attr = attr.split('.', 1)
    if len(attr) == 1:
        setattr(obj, attr[0], value)
    else:
        recursive_setattr(getattr(obj, attr[0]), attr[1], value)


@torch.no_grad()
def set_linear_module_frozen_simple(module, frozen: bool, dtype: torch.dtype, device: torch.device):
    new_module = torch.nn.Linear(module.in_features,
                                 module.out_features,
                                 module.bias is not None,
                                 module.weight.device,
                                 dtype)
    new_module.weight = torch.nn.Parameter(module.weight.detach().clone())
    new_module.bias = torch.nn.Parameter(module.bias.detach().clone()) if module.bias is not None else None
    new_module.weight.requires_grad = not frozen
    if new_module.bias is not None:
        new_module.bias.requires_grad = not frozen
    return new_module


@torch.no_grad()
def set_linear_module_frozen(module, frozen: bool, dtype: torch.dtype, device: torch.device):
    if type(module) is torch.nn.Linear:
        if frozen:
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False
            return module.to(dtype).to(device)
        else:
            new_module = ConvertingLinear.fromLinear(module).to(dtype)
            new_module.weight.requires_grad = True
            if new_module.bias is not None:
                new_module.bias.requires_grad = True
            return new_module.to(device)
    elif type(module) is ConvertingLinear:
        if not frozen:
            module.weight.requires_grad = True
            if module.bias is not None:
                module.bias.requires_grad = True
            assert False
            return module.to(dtype).to(device)
        else:
            new_module = torch.nn.utils.skip_init(torch.nn.Linear, in_features=module.in_features,
                                                  out_features=module.out_features,
                                                  bias=module.bias is not None,
                                                  device=module.weight.device,
                                                  dtype=dtype)
            new_module.weight = torch.nn.Parameter(module.weight.to(dtype))
            new_module.bias = torch.nn.Parameter(module.bias.to(dtype)) if module.bias is not None else None
            new_module.weight.requires_grad = False
            if new_module.bias is not None:
                new_module.bias.requires_grad = False
            return new_module.to(device)
    else:
        assert False


@torch.no_grad()
def freeze_random_modules(model, target_params: int, frozen_dtype: torch.dtype, frozen_device: torch.device, active_device: torch.device):
    modules = dict(model.named_modules())
    linear_names = find_all_linear_module_names(model)

    for key in linear_names:
        if modules[key].weight.dtype != frozen_dtype or modules[key].weight.requires_grad or modules[key].weight.requires_grad:
            parent, target, target_name = _get_submodules(model, key)
            setattr(parent, target_name, set_linear_module_frozen(modules[key], True, frozen_dtype, frozen_device))
    modules = dict(model.named_modules())

    active_paramter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if active_paramter_count > target_params:
        raise RuntimeError("Enough paramters must be available to train at least one linear layer")

    while active_paramter_count < target_params and len(linear_names) > 0:
        i = randint(0, len(linear_names) - 1)
        parent, target, target_name = _get_submodules(model, linear_names[i])
        new_module = set_linear_module_frozen(modules[linear_names[i]], False, torch.float32, active_device)
        setattr(parent, target_name, new_module)
        active_paramter_count += modules[linear_names[i]].weight.numel()
        if modules[linear_names[i]].bias is not None:
            active_paramter_count += modules[linear_names[i]].bias.numel()
        linear_names.pop(i)
    modules = dict()

    assert active_paramter_count == sum(p.numel() for p in model.parameters() if p.requires_grad)

    return active_paramter_count


def save_model(model, global_step: int, output_dir: str, max_checkpoints: int = 0):
    output_chkpt_dir = f"step_{global_step}" if global_step >= 0 else ""
    output_dir = os.path.join(output_dir, output_chkpt_dir)
    model.save_pretrained(output_dir)

    if max_checkpoints > 0:
        files = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)) and f.starts_with("step_")]

        def extract_step(filename):
            tokens = filename.split('_')
            return int(tokens[1])

        if len(files) > max_checkpoints:
            min_step = min(map(extract_step, extract_step))
            delete_checkpoit_dir = os.path.join(output_dir, f"step_{min_step}")
            print(f"there are more than {max_checkpoints} checkpints saved, deleting {delete_checkpoit_dir}")
            shutil.rmtree(delete_checkpoit_dir)


def get_optimizer(model, dynamic_module_names: list, static_module_names: list, lr: float, static_lr: float,
                  weight_decay: float, eps: float, adam8bit: bool):

    all_keys = dynamic_module_names + static_module_names
    duplicated = [k for k, v in collections.Counter(all_keys).items() if v > 1]
    if len(duplicated) > 0:
        print("duplicated items:")
        for item in duplicated:
            print(item)
        raise ValueError("dynamic_module_names and or static_module_names contain duplicated paramters")

    parameters = list()
    modules = dict(model.named_modules())
    for key in dynamic_module_names:
        parameters.extend({'params': p} for p in modules[key].parameters() if p.requires_grad)
    param_ids = set([id(p['params']) for p in parameters])
    for key in static_module_names:
        parameters.extend({'params': p, 'lr': static_lr} for p in modules[key].parameters() if p.requires_grad and id(p) not in param_ids)
        for p in modules[key].parameters():
            param_ids.add(id(p))

    if not adam8bit:
        optimizer = torch.optim.AdamW(parameters, weight_decay=weight_decay, lr=lr, eps=training_args.adam_epsilon)
    else:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, bitsandbytes must be available")
        optimizer = bnb.optim.AdamW8bit(parameters, weight_decay=weight_decay, lr=lr, eps=eps)
    return optimizer


def compute_dynamic_parameter_ratio(model):
    modules = dict(model.named_modules())
    active_linear_parameters = 0
    total_linear_parameters = 0
    for key in find_all_linear_module_names(model):
        active_linear_parameters += sum(p.numel() for p in modules[key].parameters() if p.requires_grad)
        total_linear_parameters += sum(p.numel() for p in modules[key].parameters())
    return math.ceil(total_linear_parameters / active_linear_parameters)


def prepare(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, primary_device: torch.device, secondary_device: torch.device) -> tuple:
    model = get_model(model_args, training_args.cache_dir, training_args.gradient_checkpointing).to(primary_device)

    tokenizer = get_tokenizer(model, training_args.cache_dir, model_args)

    if data_args.dataset.endswith("json"):
        print("Loading dataset in s2s mode")
        data_module = create_data_module_s2s(tokenizer, data_args, training_args.do_train, training_args.do_eval, False)
    else:
        print("Loading dataset in txt mode")
        data_module = create_data_module(tokenizer, data_args, training_args.do_train, training_args.do_eval, False)
    dataset = {k: v for k, v in data_module.items() if k != 'predict_dataset'}
    train_dataloader = torch.utils.data.DataLoader(
        dataset['train_dataset'],
        shuffle=True,
        collate_fn=dataset['data_collator'],
        batch_size=training_args.per_device_train_batch_size
    ) if dataset['train_dataset'] is not None else None
    eval_dataloader = torch.utils.data.DataLoader(
        dataset['eval_dataset'],
        shuffle=True,
        collate_fn=dataset['data_collator'],
        batch_size=training_args.per_device_train_batch_size
    ) if dataset['eval_dataset'] is not None else None

    if model_args.max_instant_params != 0:
        print(f"Target params {model_args.max_instant_params}m")
        freeze_random_modules(model, model_args.max_instant_params * 1e6,
                              torch.float16 if training_args.storage_fp16 else torch.float32,
                              frozen_device=primary_device, active_device=secondary_device)

    paramter_count = sum(p.numel() for p in model.parameters())
    active_paramter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training model with {paramter_count/1e6}m parameters and {active_paramter_count/1e6}m instantanous active paramters")

    dynamic_param_ratio = compute_dynamic_parameter_ratio(model)
    print(f"dyanamic parameter ratio: 1/{dynamic_param_ratio}")

    steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * training_args.epochs

    optimizer = get_optimizer(model, find_all_linear_module_names(model),
                              find_all_outher_module_names(model) if training_args.train_non_linear_layers else list(),
                              training_args.learning_rate,
                              training_args.learning_rate / dynamic_param_ratio,
                              training_args.weight_decay,
                              training_args.adam_epsilon,
                              training_args.adam8bit)
    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=total_steps
    )
    return model, optimizer, lr_scheduler, train_dataloader


def train(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    primary_device = torch.device(training_args.primary_device)
    secondary_device = torch.device(training_args.secondary_device)
    log_writer = tensorboard.SummaryWriter()

    model, optimizer, lr_scheduler, train_dataloader = prepare(model_args, data_args, training_args, primary_device, secondary_device)

    steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * training_args.epochs
    dynamic_param_ratio = compute_dynamic_parameter_ratio(model)

    if training_args.do_train:
        progress_bar = tqdm(range(total_steps))
        global_step = 0
        model.train()
        for epoch in range(0, training_args.epochs):
            print("*** Train ***")
            print(f'Vram used for model before training starts: {torch.cuda.memory_allocated()/(1024.0*1024.0)}')
            for step, batch in enumerate(train_dataloader):
                for key in batch:
                    batch[key] = batch[key].to("cuda:0")
                outputs = model(**batch)
                loss = outputs.loss / training_args.gradient_accumulation_steps
                log_writer.add_scalar("Loss/train", loss, global_step)
                loss.backward()

                if (step + 1) % training_args.gradient_accumulation_steps == 0 or step + 1 == len(train_dataloader):
                    optimizer.step()
                    lr_scheduler.step()

                    model.zero_grad()

                    if global_step % 10 == 0:
                        print(loss)

                    if global_step % 10 == 0 and model_args.max_instant_params != 0:
                        param_count = freeze_random_modules(model, model_args.max_instant_params * 1e6,
                                                            torch.float16 if training_args.storage_fp16 else torch.float32,
                                                            frozen_device=primary_device,
                                                            active_device=secondary_device)
                        log_writer.add_scalar("Parameters/train", param_count, global_step)
                        optimizer = get_optimizer(model, find_all_linear_module_names(model),
                                                  find_all_outher_module_names(model) if training_args.train_non_linear_layers else list(),
                                                  training_args.learning_rate,
                                                  training_args.learning_rate / dynamic_param_ratio,
                                                  training_args.weight_decay,
                                                  training_args.adam_epsilon,
                                                  training_args.adam8bit)
                        lr_scheduler.optimizer = optimizer

                    global_step += 1
                    progress_bar.update()

                    if global_step % training_args.save_steps == 0:
                        save_model(model, global_step, training_args.output_dir, training_args.max_checkpoints)
                if training_args.flush_allocator:
                    torch.cuda.empty_cache()

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")

    save_model(model, global_step, training_args.output_dir)

    return


if __name__ == "__main__":
    hfparser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, extra_args = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)

    print("Model Arguments:")
    print(model_args)
    print("\nData Arguments:")
    print(data_args)
    print("\nTraining Arguments:")
    print(training_args)

    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    train(model_args, data_args, training_args)
