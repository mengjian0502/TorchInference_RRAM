"""
Get weight distribution
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def binary2dec(wbit, weight_b, cellBit):
    weight_int = 0
    cellRange = 2**cellBit
    for k in range(wbit//cellBit):
        remainder = weight_b[k]
        scaler = cellRange**k
        weight_int += scaler*remainder
    return weight_int

def main():
    ckpt = torch.load("./save/vgg7_quant/vgg7_quant_w4_a4_mode_sawb_symm_wd0.0_lambda_swipe1e-4_swipe_train_th/model_best.pth.tar")
    state_dict = ckpt["state_dict"]
    
    # print weights
    # import pdb;pdb.set_trace()
    weight = state_dict['features.14.weight']
    print("Weight size = {}".format(list(weight.size())))

    from models import quant
    
    # precision
    nbit = 4
    cellBit = 1

    # quantize
    weight_q, wscale = quant.stats_quant(weight, nbit=nbit, dequantize=False)
    weight_q = weight_q.add(7)
    print("Unique levels of the {}bit weight: \n{}".format(nbit, weight_q.unique().cpu().numpy()))
    plt.figure(figsize=(8,6))
    sns.displot(weight_q.view(-1).cpu().numpy())
    plt.savefig("./save/figs/swipeTrain_4bit.png", bbox_inches = 'tight', pad_inches = 0.1)

    total_w = 0
    level_element = np.zeros(15)
    for k, v in state_dict.items():
        if len(v.size()) == 4 and v.size(1) > 3:
            wq, wscale = quant.stats_quant(v, nbit=nbit, dequantize=False)
            wq = wq.add(7)
            total_w += wq.numel()
            
            layer_element = []
            for ii in wq.unique():
                n = wq[wq==ii].numel()
                layer_element.append(n)
            print(layer_element)
            level_element += np.array(layer_element)
    perc = level_element / total_w * 100

    swipe_perc = 0
    swipe = [6,7,8]
    for ii, p in enumerate(perc):
        if ii in swipe:
            swipe_perc += p
    print("Percentage of {} = {:.2f}".format(swipe, swipe_perc))

if __name__ == '__main__':
    main()