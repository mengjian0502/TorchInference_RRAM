"""
Quantization modules
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F
from .utee import wage_quantizer

def dorefa_quant(x, nbit, dequantize=True):
    qrange = 1.0
    x = F.hardtanh(x, min_val=-qrange, max_val=qrange)
    
    scale = (2**nbit - 1) / qrange
    
    x = qrange * x / 2 / x.abs().max() + qrange/2
    xq = torch.round(x * scale)
    
    if dequantize:
        xq = xq.div(scale)
        xq = 2 * xq - qrange
    return xq

def stats_quant(x, nbit, qmode='symm', dequantize=True):
    z_typical = {'4bit': [0.077, 1.013], '8bit':[0.027, 1.114]}
    z = z_typical[f'{int(nbit)}bit']

    m = x.abs().mean()
    std = x.std()

    if qmode == 'symm':
        n_lv = 2 ** (nbit - 1) - 1
        alpha_w = 1/z[0] * std - z[1]/z[0] * m
    elif qmode == 'asymm':
        n_lv = (2 ** (nbit) - 1)/2
        alpha_w = 2*m
    else:
        raise NotImplemented

    x = x.clamp(-alpha_w.item(), alpha_w.item())
    scale = n_lv / alpha_w
    
    xq = x.mul(scale).round()
    if len(xq.unique()) > 2**nbit:
        xq = xq.clamp(-2**nbit//2, 2**nbit//2-1)
    # import pdb;pdb.set_trace()
    if dequantize:
        xq = xq.div(scale)
    return xq, scale

class RoundQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, wbit, qmode):
        input_q, scale = stats_quant(input, wbit, qmode)
        ctx.save_for_backward(input)
        return input_q
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None, None

class RoundUQ(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, alpha, nbit): 
        ctx.save_for_backward(input, alpha)

        scale = (2**nbit - 1) / alpha
        input_div = input.mul(scale)
        input_q = input_div.round().div(scale)
        return input_q

    @staticmethod
    def backward(ctx, grad_output):
        input, alpha = ctx.saved_tensors

        lower_bound = input < 0
        upper_bound = input > alpha

        x_range = ~(lower_bound|upper_bound)

        grad_alpha = torch.sum(grad_output * torch.ge(input, alpha).float()).view(-1)
        grad_input = grad_output * x_range.float()
        return grad_input, grad_alpha, None

class WQ(nn.Module):
    """
    Weight quantizer
    """
    def __init__(self, wbit, qmode='symm'):
        super(WQ, self).__init__()
        self.wbit = wbit
        self.qmode = qmode
    
    def forward(self, x):
        weight_q = RoundQ.apply(x, self.wbit, self.qmode)
        return weight_q

    def extra_repr(self):
        return super(WQ, self).extra_repr() + "qmode={}".format(self.qmode)


class AQ(nn.Module):
    def __init__(self, abit, act_alpha):
        super(AQ, self).__init__()
        self.abit = abit
        self.register_parameter('act_alpha', nn.Parameter(torch.tensor(act_alpha)))

    def forward(self, input):
        input = torch.where(input < self.act_alpha, input, self.act_alpha)
        input_q = RoundUQ.apply(input, self.act_alpha, self.abit)
        return input_q

class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, wbit=8, abit=8, alpha_init=10.0, wqmode='symm'):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups,
                                      bias=bias)
        self.weight_quant = WQ(wbit=wbit, qmode=wqmode)
        self.act_quant = AQ(abit, act_alpha=torch.tensor(alpha_init))

        self.wbit = wbit
        self.abit = abit

    def forward(self, input):
        if self.abit == 32 or self.in_channels == 3:
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
