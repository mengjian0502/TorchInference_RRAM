"""
Inference with different ADC precisions
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utee import wage_quantizer
from .quant_modules import WQ, AQ, stats_quant

def decimal2binary(weight_q, bitWeight, cellBit):
    cellRange = 2**cellBit
    remainder_list = torch.Tensor([]).type_as(weight_q)
    
    for k in range(int(bitWeight/cellBit)):
        remainder = torch.fmod(weight_q, cellRange)
        remainder = remainder.unsqueeze(0)
        remainder_list = torch.cat((remainder_list, remainder), dim=0)
        weight_q = torch.round((weight_q-remainder.squeeze(0))/cellRange)
    return remainder_list

def bit2cond(bitWeight, hrs, lrs):
    """
    Draft: replace the binary values to conductance measurement
    """
    level0 = torch.ones(bitWeight[bitWeight==0].size()).mul(hrs)
    level1 = torch.ones(bitWeight[bitWeight==1].size()).mul(lrs)

    bitWeight[bitWeight==0] = level0.cuda()
    bitWeight[bitWeight==1] = level1.cuda()

    bitWeight = bitWeight.clamp(0)
    return bitWeight

def program_noise_cond(weight_q, weight_b, hrs, lrs, swipe_ll):
    wb = torch.zeros_like(weight_b)
    weight_cond = bit2cond(weight_b, hrs, lrs)  # typical values

    for ii in range(len(weight_q.unique())):
        
        if len(weight_q.unique()) == 1:
            ii = int(weight_q.unique().item())

        idx_4b = weight_q.eq(ii)
        wb_ii = weight_cond[:, idx_4b]
        wbin_ii = weight_b[:, idx_4b]
        
        # noises
        noise = np.load(f"/home/mengjian/Desktop/ASU_research/SWIPE_analysis/prob/SWIPE/noSWIPE_25Times_raw/level{ii}_raw.npy")
        swipe = np.load(f"/home/mengjian/Desktop/ASU_research/SWIPE_analysis/prob/SWIPE/Level_4x16_SWIPE_250nPW_chip14_raw_in_16lvl/level{ii}_raw.npy")
        
        # sizes
        _, numel = wb_ii.size()
        
        bit_idx = np.arange(noise.shape[0])
        random_idx = np.random.choice(bit_idx, size=(numel))
        
        bit_random_noise = noise[random_idx, :].T
        swipe_random_noise = swipe[random_idx, :].T

        wb_cond = torch.from_numpy(bit_random_noise).float()
        swipe_cond = torch.from_numpy(swipe_random_noise).float()

        wb_cond = torch.flip(wb_cond, dims=[0])
        swipe_cond = torch.flip(swipe_cond, dims=[0])

        if not ii in swipe_ll:
            wb[:, idx_4b] = swipe_cond.cuda()
        else:
            wb[:, idx_4b] = wb_cond.cuda()
        
        # # statistics
        # hrs = wb_cond[wb_ii == 0]
        # lrs = wb_cond[wb_ii == 1]
        # print("\nLevel {}; Average HRS={}; Average LRS={}; dummy={}".format(ii, hrs.mean(), lrs.mean(), dummy))
    return wb

class RRAMConv2d(nn.Conv2d):
    r"""
    NeuroSim-based RRAM inference with low precision weights and activations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                wl_input=8, wl_weight=8, subArray=128, inference=0, cellBit=1, ADCprecision=5, swipe_ll=0):
        super(RRAMConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.wl_input = wl_input
        self.inference = inference
        self.wl_weight = wl_weight
        self.cellBit = cellBit
        self.ADCprecision = ADCprecision
        self.layer_idx = 0
        self.iter = 0
        self.init = True
        self.subArray = subArray
        
        # quantization
        self.wbit = wl_weight
        self.abit = wl_input 
        self.weight_quant = WQ(wbit=wl_weight)
        self.act_quant = AQ(abit=wl_input, act_alpha=torch.tensor(10.0))

        # conductance
        self.hrs = 1e-6
        self.lrs = 1.66e-04
        self.nonideal_unit = self.lrs - self.hrs
        self.swipe_ll = swipe_ll
    
    def _act_quant(self, input):
        act_alpha = self.act_quant.act_alpha 
        input = torch.where(input < act_alpha, input, act_alpha)

        with torch.no_grad():
            scale = (2**self.abit - 1) / act_alpha
        
        input_div = input.mul(scale)
        input_q = input_div.round()
        return input_q, scale

    def forward(self, input: Tensor) -> Tensor:        
        # quantization
        wq, w_scale = stats_quant(self.weight.data, nbit=self.wbit, dequantize=False)
        wq = wq.add(2 ** (self.wbit - 1) - 1)
        wd = torch.ones_like(wq).mul(2 ** (self.wbit - 1) - 1)
        
        # decomposition
        wqb_list = decimal2binary(wq, bitWeight=self.wbit, cellBit=self.cellBit)
        wdb_list = decimal2binary(wd, bitWeight=self.wbit, cellBit=self.cellBit)

        wqb_list = program_noise_cond(wq, wqb_list, hrs=self.hrs, lrs=self.lrs, swipe_ll=self.swipe_ll)
        wdb_list = program_noise_cond(wd, wdb_list, hrs=self.hrs, lrs=self.lrs, swipe_ll=self.swipe_ll)

        # input quantization
        xq, x_scale = self._act_quant(input)
        cellRange = 2**self.cellBit

        # targeted output size
        odim = math.floor((xq.size(2) + 2*self.padding[0] - self.dilation[0] * (wq.size(2)-1)-1)/self.stride[0] + 1)
        output = torch.zeros((xq.size(0), wq.size(0), odim, odim)).cuda()
        for i in range(wq.size(2)):
            for j in range(wq.size(3)):
                numSubArray = wq.shape[1] // self.subArray
                
                if numSubArray == 0:
                    mask = torch.zeros_like(wq)
                    mask[:,:,i,j] = 1
                    xq, x_scale = self._act_quant(input)
                    outputIN = torch.zeros_like(output)
                    xb_list = []
                    for z in range(int(self.abit)):
                        xb = torch.fmod(xq, 2)
                        xq = torch.round((xq-xb)/2)
                        macs = torch.zeros_like(output).cuda()

                        xb_list.append(xb)
                        for k in range(int(self.wbit/self.cellBit)):
                            wqb = wqb_list[k]
                            wdb = wdb_list[k]

                            outputPartial = F.conv2d(xb, wqb*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                            outputOffset = F.conv2d(xb, wdb*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                            scaler = cellRange**k

                            maci = outputPartial - outputOffset
                            maci = maci.div(self.nonideal_unit)

                            macs = macs + maci * scaler
                        
                        scalerIN = 2**z
                        outputIN = outputIN + macs * scalerIN
                    output = output + outputIN / x_scale 
                else:
                    xq, x_scale = self._act_quant(input)
                    outputIN = torch.zeros_like(output)
                    for z in range(int(self.abit)):
                        xb = torch.fmod(xq, 2)
                        xq = torch.round((xq-xb)/2)
                        total_macs = torch.zeros_like(output)

                        for s in range(numSubArray):
                            mask = torch.zeros_like(wq)
                            mask[:,(s*self.subArray):(s+1)*self.subArray, i, j] = 1
                            macs = torch.zeros_like(output).cuda()

                            for k in range(int(self.wbit/self.cellBit)):
                                wqb = wqb_list[k]
                                wdb = wdb_list[k]

                                outputPartial = F.conv2d(xb, wqb*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputOffset = F.conv2d(xb, wdb*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                scaler = cellRange**k

                                maci = outputPartial - outputOffset
                                # ADC
                                maci = wage_quantizer.LinearQuantizeOut(maci, bit=self.ADCprecision, lb=maci.min(), ub=maci.max())
                                maci = maci.div(self.nonideal_unit)
                                macs = macs + maci * scaler
                                
                            total_macs = total_macs.add(macs)
                        scalerIN = 2**z
                        outputIN = outputIN + total_macs * scalerIN
                    output = output + outputIN / x_scale 
        output = output / w_scale
        return output