import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
import math as m
torch.manual_seed(5000)

RC_dict = {25:0, 55:1, 85:2, 120:3}
RT_dict = {20:0, 100:1, 200:2, 500:3, 1000:4, 2000:5, 5000:6, 10000:7}

AVG_SLOPE = {0:-7.94e-5, 1:2.52e-3, 2:1.54e-3, 3:3.74e-4}
AVG_INT = {0:2.83e-7, 1:-8.41e-6, 2:-5.17e-6, 3:-1.25e-6}

VAR_SLOPE = {0:-2.61e-4, 1:-0.00182, 2:-0.0017, 3:-3.39e-4}
VAR_INT = {0:8.32e-7, 1:6.34e-6, 2:5.73e-6, 3:1.02e-6}

def shift(x):
    #TODO: edge case, when x contains 0
    return 2.**torch.round(torch.log2(x))

def S(bits):
    return 2.**(bits-1)

def SR(x):
    r = torch.cuda.FloatTensor(*x.size()).uniform_()
    return torch.floor(x+r)

def C(x, bits):
    if bits > 15 or bits == 1:
        delta = 0
    else:
        delta = 1. / S(bits)
    upper = 1  - delta
    lower = -1 + delta
    return torch.clamp(x, lower, upper)

def Q(x, bits):
    assert bits != -1
    if bits==1:
        return torch.sign(x)
    if bits > 15:
        return x
    return torch.round(x*S(bits))/S(bits)

def QW(x, bits, scale=1.0):
    y = Q(C(x, bits), bits)
    # per layer scaling
    if scale>1.8: y /= scale
    return y

def QE(x, bits):
    max_entry = x.abs().max()
    assert max_entry != 0, "QE blow"
    #if max_entry != 0:
    x /= shift(max_entry)
    return Q(C(x, bits), bits)

def QG(x, bits_G, lr):
    max_entry = x.abs().max()
    assert max_entry != 0, "QG blow"
    #if max_entry != 0:
    x /= shift(max_entry)
    norm = lr * x
    norm = SR(norm)
    return norm / S(bits_G)

def Retention(x, t, RC):
    """
    Original retention implementation by Xiaochen
    """
    lower = x.min().item()
    upper = x.max().item()

    slope1 = 3.7e-4
    intercept1 = -1.25e-6

    slope2 = 1.54e-3
    intercept2 = -5.17e-6

    slope3 = 2.52e-3
    intercept3 = -8.41e-6

    slope4 = -7.94e-5
    intercept4 = 2.83e-7
    T = RC + 273.15
    
    a1 = slope1 * (1/T) + intercept1
    a2 = slope2 * (1/T) + intercept2
    a3 = slope3 * (1/T) + intercept3
    a4 = slope4 * (1/T) + intercept4

    delta = torch.zeros_like(x)
    delta = torch.where((-0.5<=x) & (x<0.5),torch.ones_like(x)*(a1*m.log10(t)/(4.5e-5)),delta)
    delta = torch.where((0.5<=x) & (x<1.5),torch.ones_like(x)*(a2*m.log10(t)/(4.5e-5)),delta)
    delta = torch.where((1.5<=x) & (x<2.5),torch.ones_like(x)*(a3*m.log10(t)/(4.5e-5)),delta)
    delta = torch.where((2.5<=x) & (x<3.5),torch.ones_like(x)*(a4*m.log10(t)/(4.5e-5)),delta)

    return torch.clamp((x+delta), lower, upper)

def mean_reg(RC):
    T = RC + 273.15
    a1 = AVG_SLOPE[0] * (1/T) + AVG_INT[0]
    a2 = AVG_SLOPE[1] * (1/T) + AVG_INT[1]
    a3 = AVG_SLOPE[2] * (1/T) + AVG_INT[2]
    a4 = AVG_SLOPE[3] * (1/T) + AVG_INT[3]
    return torch.tensor([a1, a2, a3, a4])

def var_reg(RC):
    T = RC + 273.15
    eps = 1e-6
    b1 = max(VAR_SLOPE[0] * (1/T) + VAR_INT[0], eps)
    b2 = max(VAR_SLOPE[1] * (1/T) + VAR_INT[1], eps)
    b3 = max(VAR_SLOPE[2] * (1/T) + VAR_INT[2], eps)
    b4 = max(VAR_SLOPE[3] * (1/T) + VAR_INT[3], eps)
    return torch.tensor([b1, b2, b3, b4])

def RetentionAvg(x, RT, RC):
    lower = torch.min(x).item()
    upper = torch.max(x).item()
    a = mean_reg(RC)

    nonideal_unit = 4.5288663e-05

    delta = torch.zeros_like(x)
    delta = torch.where((-0.5<=x) & (x<0.5),torch.ones_like(x)*(a[0]*m.log10(RT)/(nonideal_unit)),delta)
    delta = torch.where((0.5<=x) & (x<1.5),torch.ones_like(x)*(a[1]*m.log10(RT)/(nonideal_unit)),delta)
    delta = torch.where((1.5<=x) & (x<2.5),torch.ones_like(x)*(a[2]*m.log10(RT)/(nonideal_unit)),delta)
    delta = torch.where((2.5<=x) & (x<3.5),torch.ones_like(x)*(a[3]*m.log10(RT)/(nonideal_unit)),delta)

    out = torch.clamp((x+delta), lower, upper)
    # import pdb;pdb.set_trace()    
    return out

def RetentionAll(x, RT, RC):
    lower = torch.min(x).item()
    upper = torch.max(x).item()

    logt = m.log10(RT)
    a = mean_reg(RC)
    b = var_reg(RC)
    
    delta_mu = a*logt
    delta_std = b*logt
    nonideal_unit = 4.5288663e-05   # non ideal unit was computed at 25C 20sec
    
    delta = torch.zeros_like(x)
    x_ = x.clone()
    level0_delta = torch.empty(x_[x==0].size()).normal_(delta_mu[0], delta_std[0]) / nonideal_unit
    level1_delta = torch.empty(x_[x==1].size()).normal_(delta_mu[1], delta_std[1]) / nonideal_unit
    level2_delta = torch.empty(x_[x==2].size()).normal_(delta_mu[2], delta_std[2]) / nonideal_unit
    level3_delta = torch.empty(x_[x==3].size()).normal_(delta_mu[3], delta_std[3]) / nonideal_unit

    delta[x == 0] = level0_delta.cuda()
    delta[x == 1] = level1_delta.cuda()
    delta[x == 2] = level2_delta.cuda()
    delta[x == 3] = level3_delta.cuda()
    
    out = torch.clamp((x+delta), lower, upper)
    return out

def RetentionReal(x, RT, RC):
    lower = torch.min(x).item()
    upper = torch.max(x).item()

    cond_mean = torch.load('./prob/cond_mean_measured.pt')
    cond_var = torch.load('./prob/cond_var_measured.pt')

    RT_idx = RT_dict[RT]
    RC_idx = RC_dict[RC]

    mu = cond_mean[:, RC_idx, RT_idx]
    var = cond_var[:, RC_idx, RT_idx]

    mu_init = cond_mean[:, 0, 0]
    var_init = cond_var[:, 0, 0]

    nonideal_unit = 4.5288663e-05   # non ideal unit was computed at 25C 20sec

    delta = torch.zeros_like(x)
    d_mu = mu - mu_init
    d_var = torch.abs(var - var_init)

    x_ = x.clone()
    level0_delta = torch.empty(x_[x==0].size()).normal_(d_mu[0], d_var[0]) / nonideal_unit
    level1_delta = torch.empty(x_[x==1].size()).normal_(d_mu[1], d_var[1]) / nonideal_unit
    level2_delta = torch.empty(x_[x==2].size()).normal_(d_mu[2], d_var[2]) / nonideal_unit
    level3_delta = torch.empty(x_[x==3].size()).normal_(d_mu[3], d_var[3]) / nonideal_unit

    delta[x == 0] = level0_delta.cuda()
    delta[x == 1] = level1_delta.cuda()
    delta[x == 2] = level2_delta.cuda()
    delta[x == 3] = level3_delta.cuda()
    # import pdb;pdb.set_trace()
    out = torch.clamp((x+delta), lower, upper)
    return out

def binary_retention(x, RT, RC):
    lower = torch.min(x).item()
    upper = torch.max(x).item()

    cond_mean = torch.load('./prob/cond_mean.pt')
    cond_var = torch.load('./prob/cond_var_measured.pt')

    RT_idx = RT_dict[RT]
    RC_idx = RC_dict[RC]

    avg_cond = cond_mean[:, RC_idx, RT_idx]
    var_cond = cond_var[:, RC_idx, RT_idx]

    diff = np.diff(avg_cond)
    nonideal_unit = diff.mean()

    out = x.clone()

    level0_dist = torch.empty(out[x==-1].size()).normal_(avg_cond[0], var_cond[0]) / nonideal_unit - 1 
    level1_dist = torch.empty(out[x==1].size()).normal_(avg_cond[1], var_cond[1]) / nonideal_unit

    out[x==-1] = level0_dist.cuda()
    out[x == 1] = level1_dist.cuda()
    out = out.clamp(lower, upper)
    
    return out

def dist_retention(x, RT, RC):
    lower = torch.min(x).item()
    upper = torch.max(x).item()

    cond_mean = torch.load('./prob/cond_mean_measured.pt')
    cond_var = torch.load('./prob/cond_var_measured.pt')

    RT_idx = RT_dict[RT]
    RC_idx = RC_dict[RC]

    avg_cond = cond_mean[:, RC_idx, RT_idx]
    var_cond = cond_var[:, RC_idx, RT_idx]

    # diff = avg_cond[3] - avg_cond[0]
    nonideal_unit = 4.5288663e-05   # non ideal unit was computed at 25C 20sec

    out = x.clone()
    
    level0_dist = torch.empty(out[x==0].size()).normal_(avg_cond[0], var_cond[0]) / nonideal_unit
    level1_dist = torch.empty(out[x==1].size()).normal_(avg_cond[1], var_cond[1]) / nonideal_unit
    level2_dist = torch.empty(out[x==2].size()).normal_(avg_cond[2], var_cond[2]) / nonideal_unit
    level3_dist = torch.empty(out[x==3].size()).normal_(avg_cond[3], var_cond[3]) / nonideal_unit

    # level0_dist = torch.ones(out[x==0].size()).mul_(avg_cond[0]) / nonideal_unit
    # level1_dist = torch.ones(out[x==1].size()).mul_(avg_cond[1]) / nonideal_unit
    # level2_dist = torch.ones(out[x==2].size()).mul_(avg_cond[2]) / nonideal_unit
    # level3_dist = torch.ones(out[x==3].size()).mul_(avg_cond[3]) / nonideal_unit

    out[x == 0] = level0_dist.cuda()
    out[x == 1] = level1_dist.cuda()
    out[x == 2] = level2_dist.cuda()
    out[x == 3] = level3_dist.cuda()

    # import pdb;pdb.set_trace()
    out = out.clamp(lower, upper)
    return out

def NonLinearQuantizeOut(x, bit):
    # minQ = torch.min(x)
    # delta = torch.max(x) - torch.min(x)
    k=7.0
    minQ = -k*x.abs().mean()
    maxQ = k*x.abs().mean() 
    delta = maxQ - minQ   
    #print(minQ)
    #print(delta)
    if (bit == 3) :
        # 3-bit ADC
        y = x.clone()
        base = torch.zeros_like(y)

        bound = np.array([0.02, 0.08, 0.12, 0.18, 0.3, 0.5, 0.7, 1])
        out = np.array([0.01, 0.05, 0.1, 0.15, 0.24, 0.4, 0.6, 0.85])

        ref = torch.from_numpy(bound).float()
        quant = torch.from_numpy(out).float()

        y = torch.where(y<(minQ+ref[0]*delta), torch.add(base,(minQ+quant[0]*delta)), y)
        y = torch.where(((minQ+ref[0]*delta)<=y) & (y<(minQ+ref[1]*delta)), torch.add(base,(minQ+quant[1]*delta)), y)
        y = torch.where(((minQ+ref[1]*delta)<=y) & (y<(minQ+ref[2]*delta)), torch.add(base,(minQ+quant[2]*delta)), y)
        y = torch.where(((minQ+ref[2]*delta)<=y) & (y<(minQ+ref[3]*delta)), torch.add(base,(minQ+quant[3]*delta)), y)
        y = torch.where(((minQ+ref[3]*delta)<=y) & (y<(minQ+ref[4]*delta)), torch.add(base,(minQ+quant[4]*delta)), y)
        y = torch.where(((minQ+ref[4]*delta)<=y) & (y<(minQ+ref[5]*delta)), torch.add(base,(minQ+quant[5]*delta)), y)
        y = torch.where(((minQ+ref[5]*delta)<=y) & (y<(minQ+ref[6]*delta)), torch.add(base,(minQ+quant[6]*delta)), y)
        y = torch.where(((minQ+ref[6]*delta)<=y) & (y<(minQ+ref[7]*delta)), torch.add(base,(minQ+quant[7]*delta)), y)
        
    elif (bit == 4):
        y = x.clone()
        # 4-bit ADC
        base = torch.zeros_like(y)
        
        # good for 2-bit cell
        bound = np.array([0.02, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.85, 1])
        out = np.array([0.01, 0.035, 0.065, 0.1, 0.14, 0.18, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.55, 0.65, 0.775, 0.925])
        
        ref = torch.from_numpy(bound).float()
        quant = torch.from_numpy(out).float()

        y = torch.where(y.data<(minQ+ref[0]*delta), torch.add(base,(minQ+quant[0]*delta)), y)
        y = torch.where(((minQ+ref[0]*delta)<=y.data) & (y.data<(minQ+ref[1]*delta)), torch.add(base,(minQ+quant[1]*delta)), y)
        y = torch.where(((minQ+ref[1]*delta)<=y.data) & (y.data<(minQ+ref[2]*delta)), torch.add(base,(minQ+quant[2]*delta)), y)
        y = torch.where(((minQ+ref[2]*delta)<=y.data) & (y.data<(minQ+ref[3]*delta)), torch.add(base,(minQ+quant[3]*delta)), y)
        y = torch.where(((minQ+ref[3]*delta)<=y.data) & (y.data<(minQ+ref[4]*delta)), torch.add(base,(minQ+quant[4]*delta)), y)
        y = torch.where(((minQ+ref[4]*delta)<=y.data) & (y.data<(minQ+ref[5]*delta)), torch.add(base,(minQ+quant[5]*delta)), y)
        y = torch.where(((minQ+ref[5]*delta)<=y.data) & (y.data<(minQ+ref[6]*delta)), torch.add(base,(minQ+quant[6]*delta)), y)
        y = torch.where(((minQ+ref[6]*delta)<=y.data) & (y.data<(minQ+ref[7]*delta)), torch.add(base,(minQ+quant[7]*delta)), y)
        y = torch.where(((minQ+ref[7]*delta)<=y.data) & (y.data<(minQ+ref[8]*delta)), torch.add(base,(minQ+quant[8]*delta)), y)
        y = torch.where(((minQ+ref[8]*delta)<=y.data) & (y.data<(minQ+ref[9]*delta)), torch.add(base,(minQ+quant[9]*delta)), y)
        y = torch.where(((minQ+ref[9]*delta)<=y.data) & (y.data<(minQ+ref[10]*delta)), torch.add(base,(minQ+quant[10]*delta)), y)
        y = torch.where(((minQ+ref[10]*delta)<=y.data) & (y.data<(minQ+ref[11]*delta)), torch.add(base,(minQ+quant[11]*delta)), y)
        y = torch.where(((minQ+ref[11]*delta)<=y.data) & (y.data<(minQ+ref[12]*delta)), torch.add(base,(minQ+quant[12]*delta)), y)
        y = torch.where(((minQ+ref[12]*delta)<=y.data) & (y.data<(minQ+ref[13]*delta)), torch.add(base,(minQ+quant[13]*delta)), y)
        y = torch.where(((minQ+ref[13]*delta)<=y.data) & (y.data<(minQ+ref[14]*delta)), torch.add(base,(minQ+quant[14]*delta)), y)
        y = torch.where(((minQ+ref[14]*delta)<=y.data) & (y.data<(minQ+ref[15]*delta)), torch.add(base,(minQ+quant[15]*delta)), y)
        
    elif (bit == 5):
        y = x.clone()
        # 5-bit ADC
        base = torch.zeros_like(y)
        
        # good for 2-bit cell
        # bound = np.array([0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.4, 0.44, 0.48, 0.52, 0.56, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1])
        # bound = np.array([0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96])
        # out = np.array([0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.36, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.67, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90, 0.94])
        bound = np.array([0.02, 0.06, 0.1, 0.14, 0.18, 0.22, 0.26, 0.30, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.68, 0.72, 0.76, 0.80, 0.84, 0.88, 0.92, 0.96])
        out = np.array([0.0, 0.04, 0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32, 0.35, 0.37, 0.39, 0.41, 0.43, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55, 0.57, 0.59, 0.61, 0.63, 0.66, 0.70, 0.74, 0.78, 0.82, 0.86, 0.90, 0.94])
        # out = np.array([0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.38, 0.42, 0.46, 0.5, 0.54, 0.58, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975])
        
        
        
        # 4-bit cell
        # bound = np.array([0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1])
        # out = np.array([0.001, 0.003, 0.007, 0.010, 0.015, 0.020, 0.030, 0.040, 0.055, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.32, 0.37, 0.42, 0.47, 0.52, 0.57, 0.62, 0.67, 0.75, 0.85, 0.95])
        
        ref = torch.from_numpy(bound).float()
        quant = torch.from_numpy(out).float()

        y = torch.where(y<(minQ+ref[0]*delta), torch.add(base,minQ+quant[0]*delta), y)
        y = torch.where(((minQ+ref[0]*delta)<=y) & (y<(minQ+ref[1]*delta)), torch.add(base,minQ+quant[1]*delta), y)
        y = torch.where(((minQ+ref[1]*delta)<=y) & (y<(minQ+ref[2]*delta)), torch.add(base,minQ+quant[2]*delta), y)
        y = torch.where(((minQ+ref[2]*delta)<=y) & (y<(minQ+ref[3]*delta)), torch.add(base,minQ+quant[3]*delta), y)
        y = torch.where(((minQ+ref[3]*delta)<=y) & (y<(minQ+ref[4]*delta)), torch.add(base,minQ+quant[4]*delta), y)
        y = torch.where(((minQ+ref[4]*delta)<=y) & (y<(minQ+ref[5]*delta)), torch.add(base,minQ+quant[5]*delta), y)
        y = torch.where(((minQ+ref[5]*delta)<=y) & (y<(minQ+ref[6]*delta)), torch.add(base,minQ+quant[6]*delta), y)
        y = torch.where(((minQ+ref[6]*delta)<=y) & (y<(minQ+ref[7]*delta)), torch.add(base,minQ+quant[7]*delta), y)
        y = torch.where(((minQ+ref[7]*delta)<=y) & (y<(minQ+ref[8]*delta)), torch.add(base,minQ+quant[8]*delta), y)
        y = torch.where(((minQ+ref[8]*delta)<=y) & (y<(minQ+ref[9]*delta)), torch.add(base,minQ+quant[9]*delta), y)
        y = torch.where(((minQ+ref[9]*delta)<=y) & (y<(minQ+ref[10]*delta)), torch.add(base,minQ+quant[10]*delta), y)
        y = torch.where(((minQ+ref[10]*delta)<=y) & (y<(minQ+ref[11]*delta)), torch.add(base,minQ+quant[11]*delta), y)
        y = torch.where(((minQ+ref[11]*delta)<=y) & (y<(minQ+ref[12]*delta)), torch.add(base,minQ+quant[12]*delta), y)
        y = torch.where(((minQ+ref[12]*delta)<=y) & (y<(minQ+ref[13]*delta)), torch.add(base,minQ+quant[13]*delta), y)
        y = torch.where(((minQ+ref[13]*delta)<=y) & (y<(minQ+ref[14]*delta)), torch.add(base,minQ+quant[14]*delta), y)
        y = torch.where(((minQ+ref[14]*delta)<=y) & (y<(minQ+ref[15]*delta)), torch.add(base,minQ+quant[15]*delta), y)
        y = torch.where(((minQ+ref[15]*delta)<=y) & (y<(minQ+ref[16]*delta)), torch.add(base,minQ+quant[16]*delta), y)
        y = torch.where(((minQ+ref[16]*delta)<=y) & (y<(minQ+ref[17]*delta)), torch.add(base,minQ+quant[17]*delta), y)
        y = torch.where(((minQ+ref[17]*delta)<=y) & (y<(minQ+ref[18]*delta)), torch.add(base,minQ+quant[18]*delta), y)
        y = torch.where(((minQ+ref[18]*delta)<=y) & (y<(minQ+ref[19]*delta)), torch.add(base,minQ+quant[19]*delta), y)
        y = torch.where(((minQ+ref[19]*delta)<=y) & (y<(minQ+ref[20]*delta)), torch.add(base,minQ+quant[20]*delta), y)
        y = torch.where(((minQ+ref[20]*delta)<=y) & (y<(minQ+ref[21]*delta)), torch.add(base,minQ+quant[21]*delta), y)
        y = torch.where(((minQ+ref[21]*delta)<=y) & (y<(minQ+ref[22]*delta)), torch.add(base,minQ+quant[22]*delta), y)
        y = torch.where(((minQ+ref[22]*delta)<=y) & (y<(minQ+ref[23]*delta)), torch.add(base,minQ+quant[23]*delta), y)
        y = torch.where(((minQ+ref[23]*delta)<=y) & (y<(minQ+ref[24]*delta)), torch.add(base,minQ+quant[24]*delta), y)
        y = torch.where(((minQ+ref[24]*delta)<=y) & (y<(minQ+ref[25]*delta)), torch.add(base,minQ+quant[25]*delta), y)
        y = torch.where(((minQ+ref[25]*delta)<=y) & (y<(minQ+ref[26]*delta)), torch.add(base,minQ+quant[26]*delta), y)
        y = torch.where(((minQ+ref[26]*delta)<=y) & (y<(minQ+ref[27]*delta)), torch.add(base,minQ+quant[27]*delta), y)
        y = torch.where(((minQ+ref[27]*delta)<=y) & (y<(minQ+ref[28]*delta)), torch.add(base,minQ+quant[28]*delta), y)
        y = torch.where(((minQ+ref[28]*delta)<=y) & (y<(minQ+ref[29]*delta)), torch.add(base,minQ+quant[29]*delta), y)
        y = torch.where(((minQ+ref[29]*delta)<=y) & (y<(minQ+ref[30]*delta)), torch.add(base,minQ+quant[30]*delta), y)
        y = torch.where(((minQ+ref[30]*delta)<=y) & (y<(minQ+ref[31]*delta)), torch.add(base,minQ+quant[31]*delta), y)
        
        
    else:
        y = x.clone()
    return y


def LinearQuantizeOut(x, bit, lb, ub):    
    minQ = lb
    # delta = torch.max(x) - torch.min(x)
    delta = ub - lb

    y = x.clone()
    
    stepSizeRatio = 2.**(-bit)
    stepSize = stepSizeRatio*delta.item()
    index = torch.clamp(torch.round((x-minQ.item())/stepSize), 0, (2.**(bit)-1))
    y = index*stepSize + minQ.item()

    if x.mean() == 0 and x.std() == 0:
        return x
    else:
        return y


class WAGERounding(Function):
    @staticmethod
    def forward(self, x, bits_A, bits_E, optional):
        self.optional = optional
        self.bits_E = bits_E
        self.save_for_backward(x)
        if bits_A == -1: ret = x
        else: ret = Q(x, bits_A)

        return ret

    @staticmethod
    def backward(self, grad_output):
        if self.bits_E == -1: return grad_output, None, None, None

        if self.needs_input_grad[0]:
            try:
                grad_input = QE(grad_output, self.bits_E)
            except AssertionError as e:
                print("="*80)
                print("Error backward:%s"%self.optional)
                print("-"*80)
                print(grad_output.max())
                print(grad_output.min())
                print("="*80)
                raise e
        else:
            grad_input = grad_output

        return grad_input, None, None, None

class WAGERounding_forward(Function):
    @staticmethod
    def forward(self, x, bits_A, bits_E, optional):
        self.optional = optional
        self.bits_E = bits_E
        self.save_for_backward(x)
        if bits_A == -1: ret = x
        else: ret = Q(x, bits_A)

        return ret

    @staticmethod
    def backward(self, grad_output):
        return grad_output, None, None, None


quantize_wage = WAGERounding.apply

class WAGEQuantizer(Module):
    def __init__(self, bits_A, bits_E, name="", writer=None):
        super(WAGEQuantizer, self).__init__()
        self.bits_A = bits_A
        self.bits_E = bits_E
        self.name = name
        self.writer = writer

    def forward(self, x):
        if self.bits_A != -1:
            x = C(x, self.bits_A) #  keeps the gradients
        #print(x.std())
        y = quantize_wage(x, self.bits_A, self.bits_E, self.name)
        if self.writer is not None:
            self.writer.add_histogram(
                    "activation-before/%s"%self.name, x.clone().cpu().data.numpy())
            self.writer.add_histogram(
                    "activation-after/%s"%self.name, y.clone().cpu().data.numpy())
        return y

def WAGEQuantizer_f(x, bits_A, bits_E, name=""):
        if bits_A != -1:
            x = C(x, bits_A) #  keeps the gradients
        y = quantize_wage(x, bits_A, bits_E, name)
        return y

if __name__ == "__main__":
    import numpy as np
    np.random.seed(10)
    shape = (5,5)
    # test QG
    test_data = np.random.rand(*shape)
    r = np.random.rand(*shape)
    print(test_data*10)
    print(r*10)
    test_tensor = torch.from_numpy(test_data).float()
    rand_tensor = torch.from_numpy(r).float()
    lr = 2
    bits_W = 2
    bits_G = 8
    bits_A = 8
    bits_E = 8
    bits_R = 16
    print("="*80)
    print("Gradient")
    print("="*80)
    quant_data = QG(test_tensor, bits_G, bits_R, lr, rand_tensor).data.numpy()
    print(quant_data)
    # test QA
    print("="*80)
    print("Activation")
    print("="*80)
    quant_data = QA(test_tensor, bits_A).data.numpy()
    print(quant_data)
    # test QW
    print("="*80)
    print("Weight")
    print("="*80)
    quant_data = QW(test_tensor, bits_W, scale=16.0).data.numpy()
    print(quant_data)
    # test QW
    print("="*80)
    print("Error")
    print("="*80)
    quant_data = QE(test_tensor, bits_E).data.numpy()
    print(quant_data)

