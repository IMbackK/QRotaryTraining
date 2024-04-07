import transformers
from transformers import get_scheduler

import torch
from torch.utils import tensorboard
import os
import shutil
import math
from tqdm.auto import tqdm

from arguments import DataArguments, ModelArguments, TrainingArguments
from datamodules import create_data_module_s2s, create_data_module
from tokenizer import get_tokenizer

from dyntrainmodel import DyntrainModel


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


def get_optimizer(dyamic_parameters: list[torch.nn.parameter], static_parameters: list[torch.nn.parameter], lr: float, static_lr: float,
                  weight_decay: float, eps: float, adam8bit: bool):
    parameters = list()
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


def evaluate(model: DyntrainModel, dataloader: torch.utils.data.DataLoader) -> float:
    print("*** Eval ***")
    loss = torch.zeros((1), device="cuda:0")
    model.model.eval()
    for batch in dataloader:
        for key in batch:
            batch[key] = batch[key].to("cuda:0")
        outputs = model.model(**batch)
        loss += outputs.loss
    loss = loss / len(dataloader)
    print(f"Eval Loss {loss.item()}")


def train(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    log_writer = tensorboard.SummaryWriter()

    model = DyntrainModel(model_args.model_name_or_path, training_args.cache_dir, target_active_params=training_args.max_instant_params * 1e6,
                          reshuffle_fraction=training_args.churn_percent / 100.0, gradient_checkpointing=True, trust_remote_code=True,
                          quantize=model_args.quantize)
    devices = list(torch.device(i) for i in range(0, torch.cuda.device_count()))
    model.toDevices(devices)
    model.reshuffleActive()
    model.balanceActive()

    paramter_count = sum(p.numel() for p in model.model.parameters())
    active_paramter_count = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    static_parameter_count = model.staticParameterCount() if training_args.train_non_linear_layers else 0
    print(f"Training model with {paramter_count/1e6}m parameters and {active_paramter_count/1e6}m instantanous active paramters of which {static_parameter_count} are static")

    tokenizer = get_tokenizer(model.model, training_args.cache_dir, model_args)

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

    dynamic_param_ratio = (model.staticParameterCount() + model.dynamicParameterCount()) / model.dynamicParameterCount()
    steps_per_epoch = math.ceil(len(train_dataloader) / training_args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * training_args.epochs

    optimizer = get_optimizer(model.dynamicParameters(),
                              model.staticParameters() if training_args.train_non_linear_layers else None,
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

    if training_args.do_train:
        progress_bar = tqdm(range(total_steps))
        global_step = 0
        model.model.train()
        for epoch in range(0, training_args.epochs):
            model.model.train()
            print("*** Train ***")
            print(f'Vram used for model before training starts: {torch.cuda.memory_allocated()/(1024.0*1024.0)}')
            for step, batch in enumerate(train_dataloader):
                for key in batch:
                    batch[key] = batch[key].to("cuda:0")
                outputs = model.model(**batch)
                loss = outputs.loss / training_args.gradient_accumulation_steps
                log_writer.add_scalar("Loss/train", loss, global_step)
                loss.backward()

                if (step + 1) % training_args.gradient_accumulation_steps == 0 or step + 1 == len(train_dataloader):
                    optimizer.step()
                    lr_scheduler.step()

                    model.model.zero_grad()

                    if global_step % 5 == 0:
                        print(f"Train Loss {loss.item()}")

                    if global_step % 50 == 0 and training_args.max_instant_params != 0:
                        lr_scheduler.optimizer = None
                        del optimizer
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

                    global_step += 1
                    progress_bar.update()

                if global_step > 0:
                    if global_step % training_args.save_steps == 0:
                        save_model(model.model, global_step, training_args.output_dir, training_args.max_checkpoints)
                    if training_args.eval_steps > 0 and global_step % training_args.save_steps == 0:
                        evaluate(model, eval_dataloader)
                if training_args.flush_allocator:
                    torch.cuda.empty_cache()
            if training_args.do_eval and training_args.eval_steps == -1:
                evaluate(model, eval_dataloader)

    # Evaluation
    if training_args.do_eval:
        evaluate(model, eval_dataloader)

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
