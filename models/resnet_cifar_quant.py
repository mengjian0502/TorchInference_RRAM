"""
ResNet on CIFAR10
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .quant import ClippedReLU, int_conv2d, int_linear, DoreFa_Linear, DoreFa_Conv2d
import math

class DownsampleA(nn.Module):

  def __init__(self, nIn, nOut, stride):
    super(DownsampleA, self).__init__()
    assert stride == 2
    self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

  def forward(self, x):
    x = self.avg(x)
    return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
  expansion = 1
  """
  RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
  """
  def __init__(self, inplanes, planes, stride=1, downsample=None, wbit=4, abit=4, alpha_init=10, mode='mean', k=2):
    super(ResNetBasicblock, self).__init__()


    self.conv_a = int_conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, nbit=wbit, mode=mode, k=k)  # quantization
    self.bn_a = nn.BatchNorm2d(planes)
    self.relu1 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)    # Clipped ReLU function 4 - bits

    self.conv_b = int_conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, nbit=wbit, mode=mode, k=k)  # quantization
    self.bn_b = nn.BatchNorm2d(planes)
    self.relu2 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)    # Clipped ReLU function 4 - bits
    self.downsample = downsample

  def forward(self, x):
    residual = x

    basicblock = self.conv_a(x)
    basicblock = self.bn_a(basicblock)
    basicblock = self.relu1(basicblock)

    basicblock = self.conv_b(basicblock)
    basicblock = self.bn_b(basicblock)

    if self.downsample is not None:
      residual = self.downsample(x)
    
    return self.relu2(residual + basicblock)


class CifarResNet(nn.Module):
  """
  ResNet optimized for the Cifar dataset, as specified in
  https://arxiv.org/abs/1512.03385.pdf
  """
  def __init__(self, depth, num_classes, wbit=4, abit=4, alpha_init=10, mode='mean', k=2):
    """ Constructor
    Args:
      depth: number of layers.
      num_classes: number of classes
      base_width: base width
    """
    super(CifarResNet, self).__init__()

    block = ResNetBasicblock

    #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
    layer_blocks = (depth - 2) // 6
    self.inflate = 1
    print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))
    self.num_classes = num_classes
    self.conv_1_3x3 = int_conv2d(3, 16*self.inflate, kernel_size=3, stride=1, padding=1, bias=False, nbit=wbit, mode=mode, k=k)   # skip the push process for the first conv layer
    self.relu0 = ClippedReLU(num_bits=abit, alpha=alpha_init, inplace=True)
    self.bn_1 = nn.BatchNorm2d(16*self.inflate)

    self.inplanes = 16 * self.inflate 
    self.stage_1 = self._make_layer(block, 16*self.inflate, layer_blocks, 1, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k)
    self.stage_2 = self._make_layer(block, 32*self.inflate, layer_blocks, 2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k)
    self.stage_3 = self._make_layer(block, 64*self.inflate, layer_blocks, 2, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k)
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = int_linear(64*self.inflate, num_classes, nbit=wbit, mode=mode, k=k)                         # skip the push process for the last fc layer

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        #m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, wbit=4, abit=4, alpha_init=10, mode='mean', k=2):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        int_conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, nbit=wbit, mode=mode, k=k),
        nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k))
    self.inplanes = planes * block.expansion 
    for i in range(1, blocks):
       layers.append(block(self.inplanes, planes, wbit=wbit, abit=abit, alpha_init=alpha_init, mode=mode, k=k))
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.conv_1_3x3(x)
    x = self.relu0(self.bn_1(x))
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)


class resnet20_quant:
  base=CifarResNet
  args = list()
  kwargs = {'depth': 20}

class resnet32_quant:
  base=CifarResNet
  args = list()
  kwargs = {'depth': 32}

class resnet56_quant:
  base=CifarResNet
  args = list()
  kwargs = {'depth': 56}

