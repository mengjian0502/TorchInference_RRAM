
import math
import torch.nn as nn
from .quant import QConv2d, QLinear


def make_layers_quant(cfg, batch_norm=False, wbit=4, abit=4, alpha_init=10):
    layers = list()
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = QConv2d(in_channels, v, kernel_size=3, padding=1, bias=False, wbit=wbit, abit=abit, alpha_init=alpha_init)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    7: [128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
         512, 512, 512, 512, 'M'],
}

class VGG_quant(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False, wbit=4, abit=4, alpha_init=10):
        super(VGG_quant, self).__init__()
        self.features = make_layers_quant(cfg[depth], batch_norm, wbit=wbit, abit=abit, alpha_init=alpha_init)
        if depth == 7:
            self.classifier = nn.Sequential(
                QLinear(8192, 1024, wbit=wbit, abit=abit),
                nn.BatchNorm1d(1024),
                nn.ReLU(inplace=True),
                QLinear(1024, num_classes, wbit=wbit, abit=abit)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                QLinear(512, 512, wbit=wbit, abit=abit),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                QLinear(512, 512, wbit=wbit, abit=abit),
                nn.BatchNorm1d(512),
                nn.ReLU(True),
                QLinear(512, num_classes, wbit=wbit, abit=abit),
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class vgg7_quant:
    base = VGG_quant
    args = list()
    kwargs={'depth':7, 'batch_norm':True}

class vgg7_quant_BNFalse:
    base = VGG_quant
    args = list()
    kwargs={'depth':7, 'batch_norm':False}

class vgg16_quant:
    base = VGG_quant
    args = list()
    kwargs={'depth':16, 'batch_norm':True}
