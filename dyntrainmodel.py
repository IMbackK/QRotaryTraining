from transformers import AutoModelForCausalLM
import torch
from utils import replace_module
from modules import DynamicConvertingLinear, Linear, DynamicQantizedLinear
from random import randint
import math
from tqdm import tqdm


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

    def setFrozen(self, frozen: bool, convert: bool = True) -> None:
        for module in self.modules:
            module.setFrozen(frozen, convert)

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

    def compress(self) -> None:
        for module in self.modules:
            module.compress()

    def decompress(self) -> None:
        for module in self.modules:
            module.decompress()

    def getDistanceAndError(self) -> tuple[float, float]:
        distance_accum = torch.Tensor()
        error_accum = torch.Tensor()
        for module in self.modules:
            distance, error = module.getDistanceAndError()
            distance = distance.to("cpu")
            error = error.to("cpu")
            distance_accum = torch.cat((distance_accum, distance.reshape((distance.numel()))))
            error_accum = torch.cat((error_accum, error.reshape((error.numel()))))
        return (distance_accum, error_accum)

    def check(self) -> bool:
        for module in self.modules:
            if not module.check():
                return False
        return True


class DyntrainModel:
    def __init__(self, model_name_or_path: str, cache_dir: str, quantize: bool,
                 target_active_params: int, reshuffle_fraction: float, gradient_checkpointing: bool, trust_remote_code: bool = False):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            torch_dtype=torch.float32,
            trust_remote_code=trust_remote_code,
            device_map=None
        )
        self.target_active_params = target_active_params
        self.reshuffle_fraction = reshuffle_fraction
        if reshuffle_fraction < 0.10 or reshuffle_fraction > 1:
            raise RuntimeError("reshuffle_percent must be between 0.1 and 1.0")
        self.devices = list()
        self.inital_reshufle = True

        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.frozen_linear_groups = list()
        self.active_linear_groups = list()

        linear_group_names = DyntrainModel._getLinearGroupNames(self.model)
        for group in linear_group_names:
            for key in group:
                replace_module(self.model, key, self._getModule(key, quantize, "cuda:0", "cpu"))
            self.frozen_linear_groups.append(LinearGroup(self.model, group))
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(torch.float16)
        for group in self.frozen_linear_groups:
            group.setFrozen(True, False)

    def _getModule(self, key: str, quantize: bool, active_device: torch.device, cold_device: torch.device):
        output_dtype = torch.float16 if DyntrainModel.isModuleIn16bitOutlist(key) else torch.float32
        modules = dict(self.model.named_modules())
        if quantize:
            return DynamicQantizedLinear.fromLinear(modules[key], active_device, cold_device, output_dtype, torch.float16)
        else:
            return DynamicConvertingLinear.fromLinear(modules[key].to(torch.float16), output_dtype=output_dtype)

    def _getNonlinearNames(layer: torch.nn.Module):
        names = list()
        modules = dict(layer.named_modules())

        for key in modules.keys():
            if not isinstance(modules[key], torch.nn.Linear) and len(list(modules[key].children())) == 0 or key == "lm_head":
                names.append(key)
        return names

    def _getLinearGroupNames(layer: torch.nn.Module) -> list[list[str]]:
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

    def dynamicParameters(self) -> list:
        parameters = list()
        for group in self.frozen_linear_groups + self.active_linear_groups:
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

    def getDistanceAndErrorSample(self) -> (torch.Tensor, torch.Tensor):
        index = randint(0, len(self.active_linear_groups) - 1)
        return self.active_linear_groups[index].getDistanceAndError()

    def reshuffleActive(self):
        active_count = len(self.active_linear_groups)
        index = 0
        while len(self.active_linear_groups) > active_count * (1 - self.reshuffle_fraction):
            group = self.active_linear_groups.pop(index)
            group.setFrozen(True)
            self.frozen_linear_groups.append(group)
            assert group.check()

        params = self.activeParameterCount()

        if params >= self.target_active_params:
            RuntimeError("Insuficant active parameters to suffle active")
        while params < self.target_active_params and len(self.frozen_linear_groups) > 0:
            i = randint(0, len(self.frozen_linear_groups) - 1)
            group = self.frozen_linear_groups.pop(i)
            group.setFrozen(False)
            params += group.paramCount()
            self.active_linear_groups.append(group)
            assert group.check()
        print(math.ceil(params / 1e6))

        active_params = self.activeParameterCount()

        assert self.target_active_params * 1.3 > active_params and self.target_active_params * 0.7 < active_params

    def activeParamtersByDevice(self) -> list[int]:
        out = [0] * len(self.devices)
        for group in self.active_linear_groups:
            out[self.devices.index(group.getDevice())] += group.paramCount()
        return out

    def balanceActive(self) -> None:
        active_counts = self.activeParamtersByDevice()
        bits_per_param = list()
        for i, count in enumerate(active_counts):
            memory = torch.cuda.get_device_properties(self.devices[i]).total_memory
            if i == 0:
                memory = memory * 0.8
            bits_per_param.append(count / memory)

        max_index, max_bits_per_param = max(enumerate(active_counts), key=lambda x: x[1])
        min_index, min_bits_per_param = min(enumerate(active_counts), key=lambda x: x[1])

        for group in self.active_linear_groups:
            if group.getDevice() is self.devices[max_index]:
                memory = torch.cuda.get_device_properties(self.devices[max_index]).total_memory
                if max_index == 0:
                    memory = memory * 0.8
                swing = group.paramCount() / memory
                if max_bits_per_param - swing > min_bits_per_param + swing:
                    group.inplaceTo(device=self.devices[min_index])
                    self.balanceActive()

    def toDevices(self, devices: list[torch.device]) -> None:
        assert len(devices) > 0
        modules = dict(self.model.named_modules())
        total_memory = sum(torch.cuda.get_device_properties(d).total_memory for d in devices)
        total_memory -= torch.cuda.get_device_properties(devices[0]).total_memory * 0.2
        static_param_count = self.staticParameterCount()
        total_parameter_count = static_param_count + self.dynamicParameterCount()
        params_per_byte = total_parameter_count / float(total_memory)
        print(f"{math.floor(1/params_per_byte)} bytes available per parameter")

        self.devices = devices

        for key in DyntrainModel._getNonlinearNames(self.model):
            replace_module(self.model, key, modules[key].to(devices[0]))

        linear_groups = self.active_linear_groups + self.frozen_linear_groups

        group_index = 0
        for i, device in enumerate(devices[:-1]):
            memory = torch.cuda.get_device_properties(devices).total_memory
            if i == 0:
                memory = memory * 0.8
            params_for_device = memory * params_per_byte
            params = 0
            while params_for_device > params and group_index < len(linear_groups):
                linear_groups[group_index].inplaceTo(device=device)
                params += linear_groups[group_index].paramCount()
                group_index += 1

        while group_index < len(linear_groups):
            linear_groups[group_index].inplaceTo(device=devices[-1])
            group_index += 1

        for group in tqdm(linear_groups, desc="Perpareing layers"):
            if group.isFrozen():
                group.compress()
            else:
                group.decompress()
            assert group.check()
