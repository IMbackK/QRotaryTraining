from transformers import AutoModelForCausalLM
import torch
from utils import replace_module
from modules import ConvertingLinear, Linear
from random import randint
import math


def find_all_linear_module_names(model) -> list[str]:
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, ConvertingLinear):
            module_names.add(name)

    if 'lm_head' in module_names:  # needed for 16-bit
        module_names.remove('lm_head')
    return list(module_names)


def find_all_outher_module_names(model) -> list[str]:
    module_names = set()
    for name, module in model.named_modules():
        if not (isinstance(module, torch.nn.Linear) or isinstance(module, ConvertingLinear)):
            module_names.add(name)
    return list(module_names)


class LinearGroup:
    def __init__(self, model, group_names: list):
        self.modules = list()
        model_modules = dict(model.named_modules())
        for name in group_names:
            self.modules.append(model_modules[name])
        for module in self.modules:
            assert isinstance(module, Linear)

    def inplaceTo(self, dtype: torch.dtype = None, device: torch.device = None, output_device: torch.device = None) -> None:
        for module in self.modules:
            module.inplaceTo(dtype, device)
        self.modules[-1].setOutputDevice(output_device)

    def setFrozen(self, frozen: bool) -> None:
        for module in self.modules:
            module.setFrozen(frozen)

    def isFrozen(self) -> bool:
        return self.modules[0].isFrozen()

    def parameters(self) -> list[torch.nn.Parameter]:
        params = list()
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def paramCount(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def getDevice(self) -> torch.device:
        return self.modules[0].weight.device


class DyntrainModel:
    def __init__(self, model_name_or_path: str, cache_dir: str,
                 target_active_params: int, gradient_checkpointing: bool, trust_remote_code: bool = False):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
            device_map=None
        )
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(torch.float16)
        self.linear_groups = list()
        self.target_active_params = target_active_params

        self.devices = list()

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self._prepare()
        self.reshuffleActive()

    def _get_nonlinear_names(layer: torch.nn.Module):
        names = list()
        modules = dict(layer.named_modules())

        for key in modules.keys():
            if not isinstance(modules[key], torch.nn.Linear) and len(list(modules[key].children())) == 0 or key == "lm_head":
                names.append(key)
        return names

    def _get_linear_group_names(layer: torch.nn.Module) -> list[list[str]]:
        linear_groups = list()
        list_counter = 0
        in_sequence = False
        modules = dict(layer.named_modules())

        for key in modules.keys():
            if isinstance(modules[key], torch.nn.Linear) and key != "lm_head":
                if not in_sequence:
                    linear_groups.append(list())
                    in_sequence = True
                linear_groups[list_counter].append(key)
            elif in_sequence:
                in_sequence = False
                list_counter = list_counter + 1
        return linear_groups

    def isModuleIn16bitOutlist(key: str) -> bool:
        key = key.split('.')[-1]
        whitelist = set({
            "gate_proj",
            "up_proj",
            "q_proj",
            "k_proj",
            "v_proj"})
        return key in whitelist

    def _prepare(self) -> None:
        modules = dict(self.model.named_modules())
        linear_groups = DyntrainModel._get_linear_group_names(self.model)

        for group in linear_groups:
            for key in group:
                if DyntrainModel.isModuleIn16bitOutlist(key):
                    replace_module(self.model, key, ConvertingLinear.fromLinear(modules[key].to(torch.float16), output_dtype=torch.float16))
                else:
                    replace_module(self.model, key, ConvertingLinear.fromLinear(modules[key].to(torch.float16), output_dtype=torch.float32))
            self.linear_groups.append(LinearGroup(self.model, group))

    def dynamicParameters(self) -> list:
        parameters = list()
        for group in self.linear_groups:
            parameters.extend(group.parameters())
        return parameters

    def staticParameters(self) -> list:
        modules = dict(self.model.named_modules())
        dynamic_param_ids = set([id(p) for p in self.dynamicParameters()])
        parameters = list()
        for key in modules.keys():
            for param in modules[key].parameters():
                if id(param) not in dynamic_param_ids:
                    parameters.append(param)
        return parameters

    def dynamicParameterCount(self) -> int:
        return sum(p.numel() for p in self.dynamicParameters())

    def staticParameterCount(self) -> int:
        return sum(p.numel() for p in self.staticParameters())

    def activeParameterCount(self) -> int:
        total_params = self.dynamicParameters() + self.staticParameters()
        return sum(p.numel() for p in total_params if p.requires_grad)

    def reshuffleActive(self) -> None:
        for group in self.linear_groups:
            group.setFrozen(True)

        indecies = list(range(0, len(self.linear_groups)))
        params = self.staticParameterCount()
        while params < self.target_active_params and len(indecies) > 0:
            i = randint(0, len(indecies) - 1)
            self.linear_groups[indecies[i]].setFrozen(False)
            params += self.linear_groups[indecies[i]].paramCount()
            indecies.pop(i)
        print(math.ceil(params / 1e6))

        for group in self.linear_groups:
            if group.isFrozen():
                group.inplaceTo(dtype=torch.float16)
            else:
                group.inplaceTo(dtype=torch.float32)
        active_params = self.activeParameterCount()

        assert self.target_active_params * 1.3 > active_params and self.target_active_params * 0.7 < active_params

    def balanceActive(self) -> None:
        device_groups = list()
        for index in range(0, len(self.devices)):
            device_groups.append(list())

        for group in self.linear_groups:
            if not group.isFrozen():
                device_groups[self.devices.index(group.getDevice())].append(group)

        min_index, min_count = min(enumerate(len(grouplist) for grouplist in device_groups), key=lambda x: x[1])
        max_index, max_count = max(enumerate(len(grouplist) for grouplist in device_groups), key=lambda x: x[1])

        if max_count - 2 > min_count:
            device_groups[max_index][0].inplaceTo(device=self.devices[min_index])
            self.balanceActive()

    def toDevices(self, devices: list[torch.device]) -> None:
        assert len(devices) > 0
        modules = dict(self.model.named_modules())
        total_memory = sum(torch.cuda.get_device_properties(d).total_memory for d in devices)
        static_param_count = self.staticParameterCount()
        total_parameter_count = static_param_count + self.dynamicParameterCount()
        params_per_byte = total_parameter_count / float(total_memory)
        print(f"{math.floor(1/params_per_byte)} bytes available per parameter")

        self.devices = devices

        for key in DyntrainModel._get_nonlinear_names(self.model):
            replace_module(self.model, key, modules[key].to(devices[0]))

        group_index = 0
        for device in devices[:-1]:
            params_for_device = torch.cuda.get_device_properties(devices).total_memory * params_per_byte
            params = 0
            while params_for_device > params and group_index < len(self.linear_groups):
                self.linear_groups[group_index].inplaceTo(device=device)
                params += self.linear_groups[group_index].paramCount()
                group_index += 1

        while group_index < len(self.linear_groups):
            self.linear_groups[group_index].inplaceTo(device=devices[-1])
            group_index += 1
