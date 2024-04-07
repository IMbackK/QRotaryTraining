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
                breakpoint()
                self.compress()
            else:
                self.decompress()
                self.weightStart = torch.Tensor(self.weight).clone().detach()

    def isFrozen(self) -> bool:
        return not self.weight.requires_grad

    def inplaceTo(self, dtype: torch.dtype = None, device: torch.device = None):
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
        breakpoint()
        return self


class DynamicConvertingLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None,
                 output_dtype=None, output_device=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.output_dtype = output_dtype
        self.output_device = output_device

    @classmethod
    def fromLinear(cls, in_module: torch.nn.Linear, output_dtype, output_device=None):
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
                 output_dtype=None, compute_dtype=None, output_device=None):
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
        self.quant_type = 'nf4'

    @classmethod
    def fromLinear(cls, in_module: torch.nn.Linear, active_device: torch.device, cold_device: torch.device,
                   output_dtype=None, compute_dtype=torch.float16, output_device=None):
        new_module = cls(in_features=in_module.in_features, out_features=in_module.out_features, bias=in_module.bias is not None,
                         active_device=active_device, cold_device=cold_device, output_dtype=output_dtype,
                         compute_dtype=compute_dtype, output_device=output_device)
        new_module.weight = torch.nn.Parameter(in_module.weight.to(torch.float32).to(cold_device))
        new_module.bias = torch.nn.Parameter(in_module.bias.to(torch.float32).to(cold_device)) if new_module.bias is not None else None
        return new_module

    def compress(self) -> None:
        weight = self.weight.contiguous().to(torch.float16).cuda(self.active_device)
        self.weight_quantized, self.weight_state = bnb.functional.quantize_4bit(weight, blocksize=self.block_size,
                                                                                compress_statistics=False, quant_type=self.quant_type)
        if self.bias is not None:
            bias = self.bias.contiguous().to(torch.float16).cuda(self.active_device)
            self.bias_quantized, self.bias_state = bnb.functional.quantize_4bit(bias, blocksize=self.block_size,
                                                                                compress_statistics=False, quant_type=self.quant_type)

        weight = torch.nn.Parameter(self.weight.to(self.cold_device))
        bias = torch.nn.Parameter(self.bias.to(self.cold_device)) if self.bias is not None else None

    def decompress(self) -> None:
        if self.weight_quantized is None:
            raise RuntimeError("decompress() called in quantized stated before quantized weights are avialable")
        dtype = self.weight.dtype
        self.weight = torch.nn.Parameter(bnb.functional.dequantize_fp4(self.weight_quantized, self.weight_state).to(dtype).to(self.active_device))
        if self.bias_quantized:
            self.bias = torch.nn.Parameter(bnb.functional.dequantize_fp4(self.bias_quantized, self.bias_state).to(dtype).to(self.active_device))

    def checkDistance(self) -> tuple[float, float]:
        if self.weight_quantized is None:
            raise RuntimeError("checkDistance() called without quantized weights avialable")
        original_weight = self.weight.contiguous().to(torch.float16).cuda(self.active_device)
        quantized_original_weight, quantized_original_state = bnb.functional.quantize_4bit(original_weight,
                                                                                           blocksize=self.block_size,
                                                                                           compress_statistics=True,
                                                                                           quant_type=self.quant_type)
        dequantized_original_weight = bnb.functional.dequantize_fp4(quantized_original_weight, quantized_original_state).to(original_weight.dtype)
        dequantized_weight = bnb.functional.dequantize_fp4(self.weight_quantized, self.weight_state).to(original_weight.dtype)
        distance = (torch.linalg.vector_norm(dequantized_original_weight - dequantized_weight).to(torch.float32) / dequantized_original_weight.numel()).item()
        error = (torch.linalg.vector_norm(dequantized_original_weight - original_weight).to(torch.float32) / dequantized_original_weight.numel()).item()
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
            bias = None
            if self.bias_quantized is not None:
                bias = bnb.functional.dequantize_fp4(self.bias_quantized, self.bias_state).to(x.dtype)
            out = bnb.matmul_4bit(x, self.weight_quantized.t(), bias=bias, quant_state=self.weight_state)

            return out.to(output_device).to(output_dtype)

    def inplaceTo(self, dtype: torch.dtype = None, device: torch.device = None):
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
