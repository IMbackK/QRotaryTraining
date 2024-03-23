from peft.utils import _get_submodules
import torch


def replace_module(model, key: str, module: torch.nn.Module):
    parent, target, target_name = _get_submodules(model, key)
    setattr(parent, target_name, module)


def find_all_linear_module_names(model) -> list[str]:
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module_names.add(name)

    if 'lm_head' in module_names:  # needed for 16-bit
        module_names.remove('lm_head')
    return list(module_names)


def find_all_outher_module_names(model) -> list[str]:
    module_names = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            module_names.add(name)
    return list(module_names)
