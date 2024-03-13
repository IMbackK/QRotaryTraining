import torch


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

    def setFrozen(self, frozen: bool):
        self.weight.requires_grad = not frozen
        if self.bias is not None:
            self.bias.requires_grad = not frozen

    def isFrozen(self) -> bool:
        return not self.weight.requires_grad

    def inplaceTo(self, dtype: torch.dtype = None, device: torch.device = None):
        if dtype is not None:
            self.weight = torch.nn.Parameter(self.weight.to(dtype))
        if device is not None:
            self.weight = torch.nn.Parameter(self.weight.to(device))


class ConvertingLinear(Linear):
    def __init__(self,
                 in_features, out_features, bias=True, device=None, dtype=None,
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

    def forward(self, input: torch.Tensor):
        output_dtype = input.dtype if self.output_dtype is None else self.output_dtype
        output_device = input.device if self.output_device is None else self.output_device
        if input.device != self.weight.device:
            input = input.to(self.weight.device)
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        output = torch.nn.Linear.forward(self, input)
        if torch.isnan(output).any() or self.weight.dtype != torch.float32:
            breakpoint()
        return output.to(output_device).to(output_dtype)
