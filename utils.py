
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
