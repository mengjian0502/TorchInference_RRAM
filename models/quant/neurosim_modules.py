"""
Inference with different ADC precisions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utee import wage_quantizer
from .quant_modules import dorefa_quant, WQ, AQ

def bit_noises(weight_q, bitWeight, cellBit):
    cellRange = 2**cellBit
    remainder_list = torch.Tensor([]).type_as(weight_q)
    
    for k in range(int(bitWeight/cellBit)):
        remainder = torch.fmod(weight_q, cellRange)
        remainder = remainder.unsqueeze(0)
        remainder_list = torch.cat((remainder_list, remainder), dim=0)
        weight_q = torch.round((weight_q-remainder.squeeze(0))/cellRange)
    return remainder_list

class Qconv2dDoreFa(nn.Conv2d):
    r"""
    NeuroSim-based RRAM inference with low precision weights and activations
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, 
                wl_input=8, wl_weight=8, subArray=128, inference=0, cellBit=1, ADCprecision=5):
        super(Qconv2dDoreFa, self).__init__(in_channels, out_channels, kernel_size,
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

    
    def _act_quant(self, input):
        act_alpha = self.act_quant.act_alpha 
        input = torch.where(input < act_alpha, input, act_alpha)

        with torch.no_grad():
            scale = (2**self.abit - 1) / act_alpha
        
        input_div = input.mul(scale)
        input_q = input_div.round()
        return input_q

    def forward(self, input):
        # quantization
        weight_q = self.weight_quant(self.weight)
        weight_int = dorefa_quant(self.weight, nbit=self.wbit, dequantize=False)
        input_q = self.act_quant(input)
        
        # floating-point scaling factor
        w_scale = (2**self.wbit - 1)/2
        a_scale = (2**self.abit-1)/self.act_quant.act_alpha.item()
        output_original = F.conv2d(input_q, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        if self.inference == 1:    
            if self.init:
                bit_levels = bit_noises(weight_int, self.wbit, self.cellBit)
                torch.save(bit_levels, f'./prob/vgg8_{self.wbit}bit_dorefa/bit_levels{self.layer_idx}_ideal.pt')          # change the directory for different models
                self.init = False
                import pdb;pdb.set_trace()
            else:
                bit_levels = torch.load(f'./prob/vgg8_{self.wbit}bit_dorefa/bit_levels{self.layer_idx}_ideal.pt') 

            output = torch.zeros_like(output_original)
            cellRange = 2**self.cellBit
            
            # cell range
            upper = 1   # LRS
            lower = 0   # HRS
            
            # dummy column (Offset of the fixed point values)
            dummyP = torch.zeros_like(self.weight)
            dummyP[:,:,:,:] = (cellRange-1)*(upper+lower)/2 

            for i in range(3):
                for j in range(3):
                    numSubArray = weight_int.shape[1] // self.subArray
                    if numSubArray == 0:
                        mask = torch.zeros_like(weight_q)
                        mask[:,:,i,j] = 1
                        if weight_int.shape[1] == 3:
                            X_decimal = weight_int*mask 
                            outputDiff = torch.zeros_like(output)
                            
                            for k in range(int(self.wbit/self.cellBit)):
                                remainder = bit_levels[k]
                                remainderQ = remainder*mask
                                
                                outputPartial= F.conv2d(input, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                outputDummyPartial= F.conv2d(input, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

                                scaler = cellRange**k
                                output_diff = outputPartial - outputDummyPartial
                                outputDiff = outputDiff + (output_diff)*scaler
                            output = output + outputDiff
                        else:
                            inputQ = self._act_quant(input)
                            outputIN = torch.zeros_like(output)
                            inputQ_list = []
                            for z in range(int(self.abit)):
                                inputB = torch.fmod(inputQ, 2)
                                inputQ = torch.round((inputQ-inputB)/2)
                                
                                inputQ_list.append(inputB)

                                X_decimal = weight_int*mask 
                                outputDiff = torch.zeros_like(output)
                                remainder_list = []
                                for k in range (int(self.wbit/self.cellBit)):
                                    # remainder_ideal = torch.fmod(X_decimal, cellRange)
                                    # X_decimal = torch.round((X_decimal-remainder)/cellRange)*mask
                                    
                                    remainder = bit_levels[k]
                                    remainderQ = remainder*mask
                                    remainder_list.append(remainderQ)
                                    
                                    outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

                                    scaler = cellRange**k

                                    output_diff = outputPartial - outputDummyPartial
                                    output_diff_quant = wage_quantizer.LinearQuantizeOut(output_diff, self.ADCprecision, lb=output_diff.min(), ub=output_diff.max())
                                    outputDiff = outputDiff + (output_diff_quant)*scaler
                                scalerIN = 2**z
                                outputIN = outputIN + outputDiff * scalerIN
                            output = output + outputIN * a_scale 
                    else:
                        inputQ = self._act_quant(input)
                        outputIN = torch.zeros_like(output)
                        for z in range(int(self.abit)):
                            inputB = torch.fmod(inputQ, 2)
                            inputQ = torch.round((inputQ-inputB)/2)

                            outputDiff = torch.zeros_like(output)
                            for s in range(numSubArray):
                                mask = torch.zeros_like(weight_q)
                                mask[:,(s*self.subArray):(s+1)*self.subArray, i, j] = 1
                                
                                outputSDiff = torch.zeros_like(output)
                                remainder_list = []
                                for k in range (int(self.wbit/self.cellBit)):

                                    remainder = bit_levels[k]
                                    remainderQ = remainder*mask

                                    outputPartial= F.conv2d(inputB, remainderQ*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    outputDummyPartial= F.conv2d(inputB, dummyP*mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
                                    scaler = cellRange**k
                                    output_diff = outputPartial - outputDummyPartial
                                    output_diff_quant = wage_quantizer.LinearQuantizeOut(output_diff, self.ADCprecision, lb=output_diff.min(), ub=output_diff.max())
                                    outputSDiff = outputSDiff + (output_diff_quant)*scaler
                                outputDiff = outputDiff + outputSDiff
                            scalerIN = 2**z
                            outputIN = outputIN + outputDiff * scalerIN
                        output = output + outputIN / a_scale
            output = output/w_scale 
        else:
            output = output_original
        return output