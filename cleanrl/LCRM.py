import math
from typing import Any

import torch
from torch import Tensor
import torch.nn as nn 
import torch.nn.functional as F
import torch.nn.init as init

def make_random(input,output,n_samples,rank):
    vectorize_matrix=torch.randn(input*output,rank)
    # u, sigma, v = torch.svd(vectorize_matrix)
    matrix=vectorize_matrix.view(input, output, -1)
    matrix = torch.transpose(matrix, 0, 2)
    matrix = torch.transpose(matrix, 1, 2)
    normalized_tensor = matrix / torch.norm(matrix, dim=2, keepdim=True)
    return normalized_tensor

class Linear(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    number_c: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int,number_c:int,n_sample: int=100, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.number_c=number_c
        self.fixed_weight=nn.Parameter(make_random(in_features,out_features,n_sample,number_c),requires_grad=False)
        self.scale = nn.Parameter(torch.randn(number_c, **factory_kwargs))
        if bias:
            self.bias=True
            self.fixed_bias = nn.Parameter(torch.randn( number_c,out_features), requires_grad=False)
            self.bias_scale = nn.Parameter(torch.randn( number_c, **factory_kwargs), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.normal_(self.scale, mean=0, std=0.1)
        if self. bias is not None:
            init.normal_(self.bias_scale, mean=0, std=0.1)
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.fixed_weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     init.uniform_(self.fixed_bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None:
            return F.linear(input, (self.fixed_weight * self.scale.unsqueeze(-1).unsqueeze(-1)).sum(dim=0).T, None)
        else:
            return F.linear(input, (self.fixed_weight * self.scale.unsqueeze(-1).unsqueeze(-1)).sum(dim=0).T, bias=(self.fixed_bias*self.bias_scale.unsqueeze(1)).sum(dim=0))

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'
    
