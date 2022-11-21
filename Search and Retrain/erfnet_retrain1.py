# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from genotypes2 import PRIMITIVES
from operations2 import *


class DownsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6, 9)):
        super(PSPModule, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128, eps=1e-03),  # default：eps=1e-05
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        # prior = nn.AdaptiveAvgPool2d(output_size=size)  # output_size=(1, 2)(69.44)
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)  # bias=True(69.37)
        bn = nn.BatchNorm2d(out_features, eps=1e-03)   # eps=1e-03(69.93) eps=1e-02(68.91)  eps=1e-04(68.34) eps=2e-03(69.10)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # default：align_corners=False
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class MixedOp(nn.Module):
  def __init__(self, prun_chan, dropprob, switch_cell):
    super(MixedOp, self).__init__()
    self.mixed_dw_ops = nn.ModuleList()
    for i in range(len(switch_cell)):
        if switch_cell[i]:
            primitive = PRIMITIVES[i]
            op = OPS_RE[primitive](prun_chan, dropprob)
            self.mixed_dw_ops.append(op)
            del op

  def forward(self, x):
    out = sum(op(x) for op in self.mixed_dw_ops)
    return out


class Encoder_retrain(nn.Module):
    def __init__(self, num_classes, switch_cell, prun_chan):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        cnt = 0

        self.layers.append(DownsamplerBlock(16, 64))
        for x in range(0, 4):    # 5 times
           self.layers.append(MixedOp(prun_chan[cnt], 0.03, switch_cell[cnt]))
           cnt += 1

        self.layers.append(DownsamplerBlock(prun_chan[cnt-1][2], 128))
        for x in range(0, 9):    # 2 times
            self.layers.append(MixedOp(prun_chan[cnt], 0.3, switch_cell[cnt]))
            cnt += 1

        # Only in encoder mode:
        self.pool = nn.AvgPool2d(2, stride=2)
        self.low = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64, eps=1e-03),
                nn.ReLU(),
            )
        self.ppm = PSPModule(prun_chan[cnt-1][2], 40)
        self.conv_last = nn.Sequential(
            nn.Conv2d(64 + 128, 128, kernel_size=1, bias=False),  # (69.37)
            nn.BatchNorm2d(128, eps=1e-03),
            nn.ReLU(),
            nn.Dropout2d(0.1),
        )
        # self.output_conv = nn.Conv2d(prun_chan[-1][2], num_classes, 1, stride=1, padding=0, bias=True)
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=True):
        x_ = []
        output = self.initial_block(input)
        for index, layer in enumerate(self.layers):
            if index == 0 or index == 5:
                x_.append(output)
                output = layer(output)
            else:
                output = layer(output)
                output = F.relu(output)
        x_.append(output)

        if predict:
            output_size = x_[1].size()[2:]
            # x_[0] = nn.functional.interpolate(x_[0], output_size, mode='bilinear', align_corners=False)
            x_[0] = self.pool(x_[0])
            fusion_list = []
            fusion_list.append(nn.functional.interpolate(self.ppm(x_[2]), output_size, mode='bilinear', align_corners=False))
            fusion_list.append(self.low(torch.cat([x_[1], x_[0]], 1)))
            output = self.conv_last(torch.cat(fusion_list, 1))
            output = self.output_conv(output)

        # return output
        return F.interpolate(output, size=(int(input.size(2)), int(input.size(3))), mode='bilinear')

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()]


if __name__ == '__main__':
    switch_cell = []
    for i in range(13):
        switch_cell.append([True, True, True, True, True, True])
    model = Encoder_retrain(20, switch_cell)
    x = torch.tensor(torch.ones(4, 3, 224, 224))
    y = model(x, predict=True)
    print(model.weight_parameters())
    print(y.shape)