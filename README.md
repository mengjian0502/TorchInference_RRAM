# RRAM-based Inference with Pytorch

## Outline

- Low precision model training 
  - 4-bit VGG7 CIFAR-10 example
- Post-training RRAM inference
  - NeuroSim-based tool with faster inference speed and low-precision model. 
  - Low-precision weight decomposition.

## Introduction

### Training

The model was quantized based on PACT and DoreFa quantization algorithms. For detailed implementation, please check [here](https://github.com/mengjian0502/TorchInference_RRAM/blob/d5b2304f7ef2929a20ad44b4cdcf23590ad02ada/models/quant/quant_modules.py#L73).

The model `Qconv2d` consists both Input Feature Map quantization modules `AQ` and layer-wise weight quantization modules `WQ`. The precisions can be specified from the initialization stage of the model in `train.py` file:

`model_cfg.kwargs.update({"num_classes": num_classes, "wbit": args.wbit, "abit":args.abit, "alpha_init": args.alpha_init})`

The initial commit provided an 4-bit VGG7 model training example. The weight precision and the activation precision can be specified inside the `vgg_cifar_quant.sh`. 

**Important Note:** Before running, please change the default Python path to your own path. 

### Inference with the pretrained model

The basic mapping scheme was inherited from the original 8-bit NeuroSim V1.2 inference code. The current implementation fully supports the CUDA computation and the updated version of Pytorch(1.7.0). 

The quantization modules `WQ` and `AQ ` are embedded inside the inference layer `Qcon2dDoreFa`. For more details, please check [here](https://github.com/mengjian0502/TorchInference_RRAM/blob/d5b2304f7ef2929a20ad44b4cdcf23590ad02ada/models/quant/neurosim_modules.py#L22). 

The initial commit provided a trained 4-bit VGG7 model example. Before running the inference, please download the pre-trained 4-bit model from the following link: 

https://drive.google.com/file/d/1TqV1pSbkRJcWWLAiM-vkPj4bCLXA4XzM/view?usp=sharing

To run the inference, execute `vgg_cifar_eval.sh` in your terminal. You can specify the precision of each cell and the ADC precisions inside the script. 

### Low-precision weight decomposition

The `Qcon2dDoreFa` module will first decompose the pre-trained low precision weight into specific bit-counts before the bit-by-bit processing, from LSB to MSB. For instance, given the 4-bit weight *W* with size 128 x 128 x 3 x 3, after the [decomposition](https://github.com/mengjian0502/TorchInference_RRAM/blob/d5b2304f7ef2929a20ad44b4cdcf23590ad02ada/models/quant/neurosim_modules.py#L11),  the weight tensor will be extended to a 4 x 128 x 128 x 3 x 3 and saved as an external `.pt` file under `/prob/`.  Each 1 x 128 x 128 x 3 x 3 corresponding the different bit-levels, from LSB to MSB. The following table summarizes the inference accuracy with different cell precisions: 

| VGG7: W4/A4 | ADC Precision | Inference Acc. |
| :---------: | :-----------: | :------------: |
| SW baseline |      N/A      |     92.12%     |
| 1-bit cell  |     6-bit     |     91.92%     |
| 2-bit cell  |     6-bit     |     91.63%     |


**Example:** 4-bit Weight decomposition with 1-bit cell. 

```python
print(list(weight_int.size()))
[128, 128, 3, 3]

# 4-bit integer weight
print(weight_int[15,15,:,:].cpu().numpy())
[[7. 8. 8.]
 [8. 8. 8.]
 [8. 7. 7.]]
# After decomposition
print(list(bit_levels.size()))
[4, 128, 128, 3, 3]

print(bit_levels[0,15,15,:,:].cpu().numpy())
[[1. 0. 0.]
 [0. 0. 0.]
 [0. 1. 1.]]
print(bit_levels[1,15,15,:,:].cpu().numpy())
[[1. 0. 0.]
 [0. 0. 0.]
 [0. 1. 1.]]
print(bit_levels[2,15,15,:,:].cpu().numpy())
[[1. 0. 0.]
 [0. 0. 0.]
 [0. 1. 1.]]
print(bit_levels[3,15,15,:,:].cpu().numpy())
[[0. 1. 1.]
 [1. 1. 1.]
 [1. 0. 0.]]
```

### Reference

PACT: Parameterized Clipping Activation for Quantized Neural Networks

```latex
@article{choi2018pact,
  title={Pact: Parameterized clipping activation for quantized neural networks},
  author={Choi, Jungwook and Wang, Zhuo and Venkataramani, Swagath and Chuang, Pierce I-Jen and Srinivasan, Vijayalakshmi and Gopalakrishnan, Kailash},
  journal={arXiv preprint arXiv:1805.06085},
  year={2018}
}
```

Dorefa-Net: Training low bitwidth convolutional neural networks with low bitwidth gradients

```latex
@article{zhou2016dorefa,
  title={Dorefa-net: Training low bitwidth convolutional neural networks with low bitwidth gradients},
  author={Zhou, Shuchang and Wu, Yuxin and Ni, Zekun and Zhou, Xinyu and Wen, He and Zou, Yuheng},
  journal={arXiv preprint arXiv:1606.06160},
  year={2016}
}
```

DNN+NeuroSim: An End-to-End Benchmarking Framework for Compute-in-Memory Accelerators with Versatile Device Technologies

```latex
@inproceedings{peng2019dnn+,
  title={DNN+ NeuroSim: An end-to-end benchmarking framework for compute-in-memory accelerators with versatile device technologies},
  author={Peng, Xiaochen and Huang, Shanshi and Luo, Yandong and Sun, Xiaoyu and Yu, Shimeng},
  booktitle={2019 IEEE International Electron Devices Meeting (IEDM)},
  pages={32--5},
  year={2019},
  organization={IEEE}
}
```

