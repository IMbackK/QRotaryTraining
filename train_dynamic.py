
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

import transformers
import torch
from torch.utils import tensorboard
import os
import shutil
import math
from tqdm.auto import tqdm
import gc
import sys

from arguments import DataArguments, ModelArguments, TrainingArguments
from datamodules import get_data_loaders
from tokenizer import get_tokenizer

from dyntrainmodel import DyntrainModel


def save_model(model, global_step: int, output_dir: str, max_checkpoints: int = 0):
    output_chkpt_dir = f"step_{global_step}" if global_step >= 0 else ""
    output_dir = os.path.join(output_dir, output_chkpt_dir)

    print(f"saveing model to {output_chkpt_dir}")

    temperature = model.generation_config.temperature
    top_p = model.generation_config.top_p
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.save_pretrained(output_dir)
    model.generation_config.temperature = temperature
    model.generation_config.top_p = top_p

    if max_checkpoints > 0:
        files = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f)) and f.startswith("step_")]

        def extract_step(filename):
            tokens = filename.split('_')
            return int(tokens[1])

        if len(files) > max_checkpoints:
            min_step = min(map(extract_step, files))
            delete_checkpoit_dir = os.path.join(output_dir, f"step_{min_step}")
            print(f"there are more than {max_checkpoints} checkpints saved, deleting {delete_checkpoit_dir}")
            shutil.rmtree(delete_checkpoit_dir)


def get_optimizer(dyamic_parameters: list[torch.nn.Parameter], static_parameters: list[torch.nn.Parameter] | None, lr: float, static_lr: float,
                  weight_decay: float, eps: float, adam8bit: bool):
    parameters = list[dict]()
    parameters.extend({'params': p} for p in dyamic_parameters if p.requires_grad)
    param_ids = set([id(p['params']) for p in parameters])
    if static_parameters is not None:
        for param in static_parameters:
            if param.requires_grad and id(param) not in param_ids:
                parameters.append({'params': param, 'lr': static_lr})
                param_ids.add(id(param))

    if not adam8bit:
        optimizer = torch.optim.AdamW(parameters, weight_decay=weight_decay, lr=lr, eps=training_args.adam_epsilon)
    else:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError("To use 8-bit Adam, bitsandbytes must be available")
        optimizer = bnb.optim.AdamW8bit(parameters, weight_decay=weight_decay, lr=lr, eps=eps)
    return optimizer


def move_optimizer_param(param, device: torch.device, device_map: dict):
    if isinstance(param, torch.Tensor):
        move_device = device if device is not None else device_map[id(param)]
        assert device is not None or move_device != torch.device("cpu")
        old_device = param.device
        param.data = param.data.to(move_device)
        if param._grad is not None:
            param._grad.data = param._grad.data.to(move_device)
        if device is not None and id(param) not in device_map:
            device_map[id(param)] = old_device
            assert old_device != torch.device("cpu")
    elif isinstance(param, dict):
        for subparam in param.values():
            move_optimizer_param(subparam, device, device_map)


def suspend_optimizer(optimizer) -> dict:
    device_map = dict()
    for param in optimizer.state.values():
        move_optimizer_param(param, torch.device("cpu"), device_map)
    return device_map


def resume_optimizer(optimizer, device_map: dict):
    for param in optimizer.state.values():
        move_optimizer_param(param, None, device_map)


def evaluate(model: DyntrainModel, tokenizer,
             dataloader: torch.utils.data.DataLoader, globalstep: int,
             log_writer: tensorboard.SummaryWriter, eval_prompt: str | None = None):
    with torch.no_grad():
        loss = torch.zeros((1), device="cuda:0")
        model.model.eval()

        for batch in tqdm(dataloader, desc="Doing eval"):
            for key in batch:
                batch[key] = batch[key].to("cuda:0")
            outputs = model.model(**batch)
            loss += outputs.loss
        loss = loss / len(dataloader)
        log_writer.add_scalar("Loss/Eval", loss, globalstep)
        print(f"Eval Loss {loss.item()}")

        if eval_prompt is not None:
            input_ids = tokenizer(eval_prompt, return_tensors="pt").input_ids.to(model.devices[0])
            attention_mask = torch.ones(input_ids.shape, device=model.devices[0], requires_grad=False)
            outputs = model.model.generate(input_ids, attention_mask=attention_mask, do_sample=True, temperature=1,
                                           max_new_tokens=100, min_new_tokens=100)
            response_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"Eval generation: {response_decoded}")
            log_writer.add_text("Text/Eval", response_decoded, globalstep)
        model.model.train()


def max_vram_allocated():
    max_vram_alloc = 0
    for i in range(0, torch.cuda.device_count()):
        max_vram_alloc = max(torch.cuda.memory_allocated(i), max_vram_alloc)
    return max_vram_alloc


def min_vram_allocated():
    max_vram_alloc = sys.maxsize
    for i in range(0, torch.cuda.device_count()):
        max_vram_alloc = min(torch.cuda.memory_allocated(i), max_vram_alloc)
    return max_vram_alloc


def train(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    log_writer = tensorboard.SummaryWriter(log_dir=training_args.logging_dir)

    model = DyntrainModel(model_args.model_name_or_path, training_args.cache_dir,
                          quantize=model_args.quantize,
                          target_active_params=int(training_args.max_instant_params * 1e6),
                          train_static_params=training_args.train_non_linear_layers,
                          reshuffle_fraction=training_args.churn_percent / 100.0,
                          gradient_checkpointing=True,
                          trust_remote_code=True)
    devices = list(torch.device(i) for i in range(0, torch.cuda.device_count()))
    model.toDevices(devices)
    model.reshuffleActive()
    model.balanceActive()

    paramter_count = sum(p.numel() for p in model.model.parameters())
    active_paramter_count = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    static_parameter_count = model.staticParameterCount() if training_args.train_non_linear_layers else 0
    print(f"Training model with {paramter_count / 1e6}m parameters and {active_paramter_count / 1e6}m "
          f"instantanous active paramters of which {static_parameter_count} are static")

    tokenizer = get_tokenizer(model.model, training_args.cache_dir, model_args)

    train_dataloader, eval_dataloader = get_data_loaders(tokenizer, data_args,
                                                         training_args.per_device_train_batch_size,
                                                         training_args.per_device_eval_batch_size,
                                                         training_args.do_train, training_args.do_eval)

    dynamic_param_ratio = (model.staticParameterCount() + model.dynamicParameterCount()) / model.dynamicParameterCount()
    steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps) if train_dataloader is not None else 1
    total_steps = steps_per_epoch * training_args.epochs

    optimizer = get_optimizer(model.dynamicParameters(),
                              model.staticParameters() if training_args.train_non_linear_layers else None,
                              training_args.learning_rate,
                              training_args.learning_rate / dynamic_param_ratio,
                              training_args.weight_decay,
                              training_args.adam_epsilon,
                              training_args.adam8bit)

    lr_scheduler = transformers.get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_steps,
        num_training_steps=total_steps
    )

    if training_args.do_train:
        progress_bar = tqdm(range(total_steps))
        global_step = 0
        model.model.train()
        for epoch in range(0, training_args.epochs):
            print("*** Train ***")
            print(f'Vram used for model before training starts: {torch.cuda.memory_allocated()/(1024.0**3):.2f}')
            for step, batch in enumerate(train_dataloader):
                for key in batch:
                    batch[key] = batch[key].to("cuda:0")
                outputs = model.model(**batch)
                loss = outputs.loss / training_args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % training_args.gradient_accumulation_steps == 0 or step + 1 == len(train_dataloader):
                    if global_step % training_args.logging_steps == 0:
                        log_writer.add_scalar("Loss/train", loss, global_step)
                    optimizer.step()
                    lr_scheduler.step()

                    progress_bar.set_postfix_str(f"Loss: {loss.item():.2f} Max: {max_vram_allocated()/(1024.0**3):.2f}GB"
                                                 f" Min: {min_vram_allocated()/(1024.0**3):.2f}GB")

                    model.model.zero_grad()

                    if global_step > 0:
                        if global_step % training_args.reshufle_steps == 0 and training_args.max_instant_params != 0:
                            print("Reshuffleing")
                            lr_scheduler.optimizer = None
                            del optimizer
                            # distance, error = model.getDistanceAndErrorSample()
                            # log_writer.add_histogram("Distances/Train", distance, max_bins=50)
                            # log_writer.add_histogram("Errors/Train", error, max_bins=50)

                            model.reshuffleActive()
                            model.balanceActive()
                            log_writer.add_scalar("Parameters/train", model.activeParameterCount(), global_step)
                            optimizer = get_optimizer(model.dynamicParameters(),
                                                      model.staticParameters() if training_args.train_non_linear_layers else None,
                                                      training_args.learning_rate,
                                                      training_args.learning_rate / dynamic_param_ratio,
                                                      training_args.weight_decay,
                                                      training_args.adam_epsilon,
                                                      training_args.adam8bit)
                            lr_scheduler.optimizer = optimizer

                        if global_step % training_args.save_steps == 0:
                            save_model(model.model, global_step, training_args.output_dir, training_args.max_checkpoints)
                        if training_args.eval_steps > 0 and global_step % training_args.eval_steps == 0:
                            device_map = suspend_optimizer(optimizer)
                            evaluate(model, tokenizer, eval_dataloader, global_step, log_writer, training_args.eval_prompt)
                            resume_optimizer(optimizer, device_map)

                    global_step += 1
                    progress_bar.update()

                if training_args.flush_allocator:
                    gc.collect()
                    torch.cuda.empty_cache()
            if training_args.do_eval and training_args.eval_steps == -1:
                device_map = suspend_optimizer(optimizer)
                evaluate(model, tokenizer, eval_dataloader, global_step, log_writer, training_args.eval_prompt)
                resume_optimizer(optimizer, device_map)

    del optimizer

    if training_args.do_eval:
        evaluate(model, tokenizer, eval_dataloader, global_step, log_writer, training_args.eval_prompt)

    save_model(model.model, global_step, training_args.output_dir)

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
