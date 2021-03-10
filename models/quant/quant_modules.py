"""
Quantization modules
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from .utee import wage_quantizer

def dorefa_quant(x, nbit, dequantize=True):
    x = torch.tanh(x)
    scale = 2**nbit - 1
    
    x = x / 2 / x.abs().max() + 1/2
    xq = torch.round(x * scale)
    
    if dequantize:
        xq = xq.div(scale)
        xq = 2 * xq - 1
    return xq

class RoundQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, wbit):
        input_q = dorefa_quant(input, wbit)
        ctx.save_for_backward(input)
        return input_q
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class RoundUQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale): 
        input_div = input.mul(scale)
        input_q = input_div.round().div(scale)
        return input_q

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class WQ(nn.Module):
    """
    DoreFa quantizer
    """
    def __init__(self, wbit):
        super(WQ, self).__init__()
        self.wbit = wbit
    
    def forward(self, x):
        weight_q = RoundQ.apply(x, self.wbit)
        return weight_q

class AQ(nn.Module):
    def __init__(self, abit, act_alpha):
        super(AQ, self).__init__()
        self.abit = abit
        self.register_parameter('act_alpha', nn.Parameter(torch.tensor(act_alpha)))

    def forward(self, input):
        input = torch.where(input < self.act_alpha, input, self.act_alpha)
        
        with torch.no_grad():
            scale = (2**self.abit - 1) / self.act_alpha 

        input_q = RoundUQ.apply(input, scale)
        return input_q

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, wbit=8, abit=8, alpha_init=10.0):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups,
                                      bias=bias)
        self.weight_quant = WQ(wbit=wbit)
        self.act_quant = AQ(abit, act_alpha=torch.tensor(alpha_init))

        self.wbit = wbit
        self.abit = abit

    def forward(self, input):
        if self.abit == 32:
            input_q = input
        else:
            input_q = self.act_quant(input)
        
        weight_q = self.weight_quant(self.weight)
        
        out = F.conv2d(input_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out

    def extra_repr(self):
        return super(QConv2d, self).extra_repr() + ", wbit={}, abit={}".format(self.wbit, self.abit)


class QLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias=True, wbit=8, abit=8, alpha_init=10.0):
        super(QLinear, self).__init__(in_features=in_channels, out_features=out_channels, bias=bias)
        self.weight_quant = WQ(wbit=wbit)
        self.act_quant = AQ(abit, act_alpha=torch.tensor(alpha_init))

        self.wbit = wbit
        self.abit = abit

    def forward(self, input):
        weight_q = self.weight_quant(self.weight)
        input_q = self.act_quant(input)
        out = F.linear(input_q, weight_q, self.bias)
        return out

    def extra_repr(self):
        return super(QLinear, self).extra_repr() + ", wbit={}, abit={}".format(self.wbit, self.abit)
