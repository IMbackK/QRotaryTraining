from peft.utils import _get_submodules
import torch


def replace_module(model, key: str, module: torch.nn.Module):
    parent, target, target_name = _get_submodules(model, key)
    setattr(parent, target_name, module)
