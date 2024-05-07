import copy
import torch
import typing
import datasets
import itertools
import transformers
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence

from arguments import DataArguments

IGNORE_INDEX = -100


def group_texts(examples, block_size: int):
    # Concatenate all texts.
    concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {k: [t[i: i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()}
    result["labels"] = result["input_ids"].copy()
    return result


@dataclass
class DataCollatorForCausalLMText(object):
    tokenizer: transformers.PreTrainedTokenizer
    max_len: int

    def __call__(self, instances: typing.Sequence[typing.Dict]) -> typing.Dict[str, torch.Tensor]:
        # Extract elements
        examples = [f"{self.tokenizer.bos_token}{example['text']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_examples = self.tokenizer(
            examples,
            max_length=self.max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        for tokenized_example in tokenized_examples['input_ids']:
            input_ids.append(torch.tensor(tokenized_example))
        # Apply padding
        padding_value = None
        if self.tokenizer.pad_token_id is not None:
            padding_value = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            padding_value = self.tokenizer.eos_token_id
        else:
            raise RuntimeError("Model dose not have a pad or eos token")
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(padding_value),
            'labels': input_ids
        }
        return data_dict


@dataclass
class DataCollatorForCausalLMs2s(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: typing.Sequence[typing.Dict]) -> typing.Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        padding_value = None
        if self.tokenizer.pad_token_id is not None:
            padding_value = self.tokenizer.pad_token_id
        elif self.tokenizer.eos_token_id is not None:
            padding_value = self.tokenizer.eos_token_id
        else:
            raise RuntimeError("Model dose not have a pad or eos token")
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=padding_value)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(padding_value),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


def create_data_module_s2s(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, do_train, do_eval, do_predict) -> typing.Dict:
    try:
        dataset = datasets.Dataset.from_json(path_or_paths=data_args.dataset)
    except FileNotFoundError as ex:
        raise ValueError(f"Error loading dataset from {data_args.dataset}, {ex}")

    if do_eval or do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset.train_test_split(
                test_size=data_args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    if 'train' in dataset:
        train_dataset = dataset['train']
    else:
        train_dataset = dataset

    train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLMs2s(
        tokenizer=tokenizer,
        source_max_len=data_args.source_max_len,
        target_max_len=data_args.target_max_len,
        train_on_source=data_args.train_on_source,
        predict_with_generate=False  # args.predict_with_generate,
    )

    return dict(
        train_dataset=train_dataset if do_train else None,
        eval_dataset=eval_dataset if do_eval else None,
        predict_dataset=eval_dataset if do_predict else None,
        data_collator=data_collator
    )


def create_data_module_hub(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, do_train, do_eval, do_predict) -> typing.Dict:
    try:
        dataset = datasets.load_dataset(data_args.dataset)
    except FileNotFoundError as ex:
        raise ValueError(f"Error loading dataset from {data_args.dataset}, {ex}")

    if do_eval or do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset.train_test_split(
                test_size=data_args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']

    if 'train' in dataset:
        train_dataset = dataset['train']
    else:
        train_dataset = dataset

    data_collator = DataCollatorForCausalLMText(
        tokenizer=tokenizer,
        max_len=data_args.source_max_len,
    )

    return dict(
        train_dataset=train_dataset if do_train else None,
        eval_dataset=eval_dataset if do_eval else None,
        predict_dataset=eval_dataset if do_predict else None,
        data_collator=data_collator
    )


def create_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments, do_train: bool, do_eval: bool, do_predict: bool) -> typing.Dict:
    try:
        dataset = datasets.load_dataset('text', data_files={'train': [data_args.dataset]})
    except FileNotFoundError as ex:
        raise ValueError(f"Error loading dataset from {data_args.dataset}, {ex}")

    if data_args.block_size > tokenizer.model_max_length:
        raise ValueError(f"Block size of {data_args.block_size} is larger than the maximum size supported by the model: {tokenizer.model_max_length}")

    def add_newline_fn(example):
        example['text'] = example['text'] + '\n'
        return example
    dataset = dataset.map(add_newline_fn)

    eval_dataset = None
    if do_eval or do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset['train'].train_test_split(
                test_size=data_args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']

    if 'train' in dataset:
        train_dataset = dataset['train']
    else:
        train_dataset = dataset

    train_dataset_tokenized = train_dataset.map(
        lambda example: tokenizer(example['text']),
        batched=True,
        remove_columns='text',
        num_proc=32,
        load_from_cache_file=True)
    train_dataset_tokenized = train_dataset_tokenized.map(
        lambda example: group_texts(example, data_args.block_size),
        batched=True,
        num_proc=32,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {data_args.block_size}")

    eval_dataset_tokenized = None
    if eval_dataset is not None:
        eval_dataset_tokenized = eval_dataset.map(
            lambda example: tokenizer(example['text']),
            batched=True,
            remove_columns='text',
            num_proc=32)
        eval_dataset_tokenized = eval_dataset_tokenized.map(
            lambda example: group_texts(example, data_args.block_size),
            batched=True,
            num_proc=32,
            load_from_cache_file=True,
            desc=f"Grouping texts in chunks of {data_args.block_size}")

    return dict(
        train_dataset=train_dataset_tokenized if do_train else None,
        eval_dataset=eval_dataset_tokenized if do_eval else None,
        predict_dataset=eval_dataset_tokenized if do_predict else None,
        data_collator=transformers.default_data_collator
    )
