import torch
import bitsandbytes as bnb
import torch.multiprocessing as multiprocessing
from typing import overload, Optional, Union
from functools import wraps


class Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)

    @classmethod
    def fromLinear(cls, in_module: torch.nn.Linear):
        new_module = torch.nn.utils.skip_init(cls, in_features=in_module.in_features,
                                              out_features=in_module.out_features,
                                              bias=in_module.bias is not None,
                                              device=in_module.weight.device,
                                              dtype=in_module.weight.dtype)
        new_module.weight = in_module.weight
        new_module.bias = in_module.bias
        return new_module

    def compress(self) -> None:
        self.inplaceTo(torch.float16)

    def decompress(self) -> None:
        self.inplaceTo(torch.float32)

    def setFrozen(self, frozen: bool, convert: bool = True):
        self.weight.requires_grad = not frozen
        if self.bias is not None:
            self.bias.requires_grad = not frozen
        if convert:
            if frozen:
                self.compress()
            else:
                self.decompress()

    def isFrozen(self) -> bool:
        return not self.weight.requires_grad

    def inplaceTo(self, dtype: torch.dtype | None = None, device: torch.device | None = None):
        frozen = self.isFrozen()
        if dtype is not None:
            self.weight = torch.nn.Parameter(self.weight.to(dtype))
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias.to(dtype))
        if device is not None:
            self.weight = torch.nn.Parameter(self.weight.to(device))
            if self.bias is not None:
                self.bias = torch.nn.Parameter(self.bias.to(device))
        Linear.setFrozen(self, frozen, False)

    def _apply(self, fn, recurse: bool = True):
        if fn.__name__ == "convert":
            return self
        else:
            return super()._apply(fn, recurse)

    @wraps(torch.nn.Module.to)
    def to(self, *args, **kwargs):
        return self

    def check(self) -> bool:
        if self.isFrozen() and self.weight.dtype != torch.float16:
            return False
        elif not self.isFrozen() and self.weight.dtype != torch.float32:
            return False
        return True


class DynamicConvertingLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None,
                 output_dtype=None, output_device=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.output_dtype = output_dtype
        self.output_device = output_device

    @classmethod
    def fromLinear(cls, in_module: torch.nn.Linear, output_dtype=torch.float32, output_device=None):
        new_module = torch.nn.utils.skip_init(cls, in_features=in_module.in_features,
                                              out_features=in_module.out_features,
                                              bias=in_module.bias is not None,
                                              device=in_module.weight.device,
                                              dtype=in_module.weight.dtype)
        new_module.output_dtype = output_dtype
        new_module.output_device = output_device
        new_module.weight = in_module.weight
        new_module.bias = in_module.bias
        return new_module

    def setOutputDevice(self, output_device: torch.device):
        self.output_device = output_device

    def checkDistance(self) -> tuple[float, float]:
        return (10.0, 0.0)

    def forward(self, input: torch.Tensor):
        output_dtype = input.dtype if self.output_dtype is None else self.output_dtype
        output_device = input.device if self.output_device is None else self.output_device
        if input.device != self.weight.device:
            input = input.to(self.weight.device)
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        output = torch.nn.Linear.forward(self, input)
        return output.to(output_device).to(output_dtype)


class DynamicQantizedLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool, active_device: torch.device, cold_device: torch.device,
                 output_dtype=None, compute_dtype=None, output_device=None, cold_dtype=torch.float32):
        super().__init__(in_features, out_features, bias, cold_device, torch.float32)
        self.active_device = active_device
        self.cold_device = cold_device
        self.output_device = output_device
        self.output_dtype = output_dtype
        self.compute_dtype = compute_dtype
        self.weight_quantized = None
        self.weight_state = None
        self.bias_quantized = None
        self.bias_state = None
        self.block_size = 128
        #self.weight_start = self.weight.clone().detach()
        self.cold_dtype = cold_dtype

    @classmethod
    def fromLinear(cls, in_module: torch.nn.Linear, active_device: torch.device = torch.device("cuda:0"), cold_device: torch.device = torch.device("cpu"),
                   output_dtype=None, compute_dtype=torch.float16, output_device=None):
        new_module = cls(in_features=in_module.in_features, out_features=in_module.out_features, bias=in_module.bias is not None,
                         active_device=active_device, cold_device=cold_device, output_dtype=output_dtype,
                         compute_dtype=compute_dtype, output_device=output_device)
        new_module.weight = torch.nn.Parameter(in_module.weight.to(torch.float32).to(cold_device))
        new_module.bias = torch.nn.Parameter(in_module.bias.to(torch.float32).to(cold_device)) if new_module.bias is not None else None
        #new_module.weight_start = new_module.weight.clone().detach()
        return new_module

    def compress(self) -> None:
        weight = self.weight.contiguous().to(torch.float16).to(self.active_device)
        self.weight_quantized, self.weight_state = bnb.functional.quantize_blockwise(weight, blocksize=self.block_size)
        if self.bias is not None:
            bias = self.bias.contiguous().to(torch.float16).to(self.active_device)
            self.bias_quantized, self.bias_state = bnb.functional.quantize_blockwise(bias, blocksize=self.block_size)

        frozen = self.isFrozen()
        self.weight = torch.nn.Parameter(self.weight.to(self.cold_dtype).to(self.cold_device))
        self.bias = torch.nn.Parameter(self.bias.to(self.cold_dtype).to(self.cold_device)) if self.bias is not None else None
        self.setFrozen(frozen, False)

    def decompress(self) -> None:
        self.weight_quantized = None
        self.weight_state = None
        self.bias_quantized = None
        self.bias_state = None
        #self.weight_start = self.weight.clone().detach().to(self.cold_device)
        self.weight = torch.nn.Parameter(self.weight.to(self.active_device).to(torch.float32))
        if self.bias_quantized:
            self.bias = torch.nn.Parameter(self.bias.to(self.active_device).to(torch.float32))

    def getDistanceAndError(self) -> tuple[torch.Tensor, torch.Tensor]:
        original_weight = self.weight.contiguous().to(self.active_device).to(torch.float16)
        quantized_original_weight, quantized_original_state = bnb.functional.quantize_blockwise(original_weight, blocksize=self.block_size)
        dequantized_original_weight = bnb.functional.dequantize_blockwise(quantized_original_weight, quantized_original_state).to(original_weight.dtype)
        distance = torch.zeros((2)) #(self.weight_start - self.weight.to(self.cold_device)).to(torch.float32)
        error = (dequantized_original_weight - original_weight).to(torch.float32)
        return (distance, error)

    def setOutputDevice(self, output_device: torch.device):
        self.output_device = output_device

    def forward(self, x: torch.Tensor):
        output_dtype = x.dtype if self.output_dtype is None else self.output_dtype
        output_device = x.device if self.output_device is None else self.output_device

        if not self.isFrozen():
            if x.device != self.weight.device:
                x = x.to(self.weight.device)
            if x.dtype != self.weight.dtype:
                x = x.to(self.weight.dtype)
            return super().forward(x).to(output_device).to(output_dtype)
        else:
            if self.weight_quantized is None:
                raise RuntimeError("forward() called in quantized stated before quantized weights are avialable")
            if x.device != self.weight_quantized.device:
                x = x.to(self.weight_quantized.device)
            weight = bnb.functional.dequantize_blockwise(self.weight_quantized, self.weight_state).to(x.dtype)
            out = torch.matmul(x, weight.t())
            if self.bias_quantized is not None:
                bias = bnb.functional.dequantize_blockwise(self.bias_quantized, self.bias_state).to(x.dtype)
                out = out + bias

            if torch.isnan(out).sum().item() > 0:
                breakpoint()

            return out.to(output_device).to(output_dtype)

    def inplaceTo(self, dtype: torch.dtype | None = None, device: torch.device | None = None):
        if dtype is not None:
            super().inplaceTo(dtype=dtype)
        if device is not None:
            frozen = self.isFrozen()
            self.active_device = device
            if self.weight_quantized is not None:
                self.weight_quantized = self.weight_quantized.to(device)
                self.weight_state = self.weight_state.to(device)
                if self.bias_quantized is not None:
                    self.bias_quantized = self.bias_quantized.to(device)
                    self.bias_state = self.bias_state.to(device)
            if not frozen:
                super().inplaceTo(device=device)
            self.setFrozen(frozen, False)

    def check(self) -> bool:
        if self.isFrozen():
            if torch.device(self.weight.device) != torch.device(self.cold_device):
                breakpoint()
                print("Frozen but not cold")
                return False
            if self.weight_quantized is None:
                breakpoint()
                print("Frozen but not quanted")
                return False
        else:
            if torch.device(self.weight.device) != torch.device(self.active_device):
                breakpoint()
                print("Active but not warm")
                return False
            if self.weight_quantized is not None:
                breakpoint()
                print("Active but still quantized")
                return False
        return True
