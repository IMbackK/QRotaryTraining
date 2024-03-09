import torch


class ConvertingLinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, output_dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.output_dtype = output_dtype

    def forward(self, input: torch.Tensor):
        output_dtype = input.dtype if self.output_dtype is None else self.output_dtype
        output_device = input.device
        if input.device != self.weight.device:
            input = input.to(self.weight.device)
        if input.dtype != self.weight.dtype:
            input = input.to(self.weight.dtype)
        output = torch.nn.Linear.forward(self, input)
        if torch.isnan(output).any() or self.weight.dtype != torch.float32:
            breakpoint()
        return output.to(output_device).to(output_dtype)

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
