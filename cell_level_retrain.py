import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from operations1 import *
from torch.autograd import Variable
from genotypes1 import PRIMITIVES


def _SplitChannels(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class MixedOp(nn.Module):
  def __init__(self, C, stride, switch_cell):
    super(MixedOp, self).__init__()
    self.mixed_dw_ops = nn.ModuleList()
    for i in range(len(switch_cell)):
        if switch_cell[i]:
            primitive = PRIMITIVES[i]
            # op = OPS[primitive](C, stride, False)
            op = OPS[primitive](C, C, stride, False, width_mult_list=[1.])
            self.mixed_dw_ops.append(op)
            del op

  def forward(self, x):
    out = sum(op(x) for op in self.mixed_dw_ops)
    return out


class Cell(nn.Module):
    def __init__(self, group_multiplier, prev_fmultiplier_down, prev_fmultiplier_same, prev_fmultiplier_up,
                 filter_multiplier, switch_cell):
        super(Cell, self).__init__()

        self.C_in = filter_multiplier * group_multiplier
        if prev_fmultiplier_down is not None:
            self.C_prev_down = int(prev_fmultiplier_down * group_multiplier)
            self.preprocess_down = ConvNorm(
                self.C_prev_down, self.C_in, 1, 1, 0, bias=True, groups=1, slimmable=False)
        if prev_fmultiplier_same is not None:
            self.C_prev_same = int(prev_fmultiplier_same * group_multiplier)
            self.preprocess_same = ConvNorm(
                self.C_prev_same, self.C_in, 1, 1, 0, bias=True, groups=1, slimmable=False)
        if prev_fmultiplier_up is not None:
            self.C_prev_up = int(prev_fmultiplier_up * group_multiplier)
            self.preprocess_up = ConvNorm(
                self.C_prev_up, self.C_in, 1, 1, 0, bias=True, groups=1, slimmable=False)

        self.group_multiplier = group_multiplier
        self._ops = MixedOp(self.C_in, 1, switch_cell)

        # self._initialize_weights()

    def scale_dimension(self, dim, scale):
        assert isinstance(dim, int)
        # if scale == 0.5:
        #     return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)
        # return int(dim * scale)
        return int((float(dim) - 1.0) * scale + 1.0) if dim % 2 else int(dim * scale)

    def prev_feature_resize(self, prev_feature, mode):
        if mode == 'down':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 0.5)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 0.5)
        elif mode == 'up':
            feature_size_h = self.scale_dimension(prev_feature.shape[2], 2)
            feature_size_w = self.scale_dimension(prev_feature.shape[3], 2)

        return F.interpolate(prev_feature, (feature_size_h, feature_size_w), mode='bilinear')

    def forward(self, s1_down, s1_same, s1_up):
        final_concates = []
        if s1_down is not None:
            s1_down = self.prev_feature_resize(s1_down, 'down')
            s1_down = self.preprocess_down(s1_down)
            final_concates.append(self._ops(s1_down))
            del s1_down
        if s1_same is not None:
            s1_same = self.preprocess_same(s1_same)
            final_concates.append(self._ops(s1_same))
            del s1_same
        if s1_up is not None:
            s1_up = self.prev_feature_resize(s1_up, 'up')
            s1_up = self.preprocess_up(s1_up)
            final_concates.append(self._ops(s1_up))
            del s1_up

        return final_concates

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

