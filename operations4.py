import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from thop import profile

class non_bottleneck_1d_rn(nn.Module):
    def __init__(self, C_in, C_out, stride, dropprob, dlt, group_width=16):
        super().__init__()
        # bottleneck_width = int(chann/2)
        cardinality = int(C_out/group_width)
        self.conv1x1_1 = nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=True)
        self.bn1 = nn.BatchNorm2d(C_out, eps=1e-03)

        self.conv3x1 = nn.Conv2d(C_out, C_out, (3, 1), stride=stride, padding=(dlt, 0),
                                 groups=cardinality, bias=True, dilation=(dlt, 1))
        self.conv1x3 = nn.Conv2d(C_out, C_out, (1, 3), stride=1, padding=(0, dlt),
                                 groups=cardinality, bias=True, dilation=(1, dlt))
        self.bn = nn.BatchNorm2d(C_out, eps=1e-03)

        self.conv1x1_2 = nn.Conv2d(C_out, C_out, 1, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(C_out, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)
        self.shortcut = nn.Sequential()
        if stride != 1 or C_in != C_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(C_out, eps=1e-03)
            )

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1x1_1(input)))

        output = F.relu(self.conv3x1(output))
        output = F.relu(self.bn(self.conv1x3(output)))

        output = self.bn2(self.conv1x1_2(output))

        if (self.dropout.p != 0):
            output = self.dropout(output)

        # return F.relu(output + input)  # +input = identity (residual connection)
        return output + self.shortcut(input)


class non_bottleneck_1d_r2(nn.Module):
    def __init__(self, chann, dropprob):
        super().__init__()

        self.conv3x1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 2, 0), bias=True, dilation=(2, 1))
        self.conv1x3 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 2), bias=True, dilation=(1, 2))
        self.bn = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1(input)
        output = F.relu(output)
        output = self.conv1x3(output)
        output = self.bn(output)

        # output = output1+output2
        if (self.dropout.p != 0):
            output = self.dropout(output)

        # return F.relu(output + input)  # +input = identity (residual connection)
        return output + input   # +input = identity (residual connection)

class non_bottleneck_1d_r3(nn.Module):
    def __init__(self, chann, dropprob):
        super().__init__()

        self.conv3x1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 4, 0), bias=True, dilation=(4, 1))
        self.conv1x3 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 4), bias=True, dilation=(1, 4))
        self.bn = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1(input)
        output = F.relu(output)
        output = self.conv1x3(output)
        output = self.bn(output)

        # output = output1+output2+output3
        if (self.dropout.p != 0):
            output = self.dropout(output)

        # return F.relu(output + input)  # +input = identity (residual connection)
        return output + input   # +input = identity (residual connection)

class non_bottleneck_1d_r4(nn.Module):
    def __init__(self, chann, dropprob):
        super().__init__()

        self.conv3x1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 8, 0), bias=True, dilation=(8, 1))
        self.conv1x3 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 8), bias=True, dilation=(1, 8))
        self.bn = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1(input)
        output = F.relu(output)
        output = self.conv1x3(output)
        output = self.bn(output)

        # output = output1+output2+output3+output4
        if (self.dropout.p != 0):
            output = self.dropout(output)

        # return F.relu(output + input)  # +input = identity (residual connection)
        return output + input   # +input = identity (residual connection)

class non_bottleneck_1d_r5(nn.Module):
    def __init__(self, chann, dropprob):
        super().__init__()

        self.conv3x1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 16, 0), bias=True, dilation=(16, 1))
        self.conv1x3 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 16), bias=True, dilation=(1, 16))
        self.bn = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1(input)
        output = F.relu(output)
        output = self.conv1x3(output)
        output = self.bn(output)

        # output = output1+output2+output3+output4+output5
        if (self.dropout.p != 0):
            output = self.dropout(output)

        # return F.relu(output + input)  # +input = identity (residual connection)
        return output + input  # +input = identity (residual connection)

class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
                                   dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
                                   dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        # return F.relu(output + input)  # +input = identity (residual connection)
        return output + input  # +input = identity (residual connection)


BatchNorm2d = nn.BatchNorm2d


class ConvNorm(nn.Module):
    '''
    conv => norm => activation
    use native nn.Conv2d, not slimmable
    '''
    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False, slimmable=True, width_mult_list=[1.]):
        super(ConvNorm, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride
        if padding is None:
            # assume h_out = h_in / s
            self.padding = int(np.ceil((dilation * (kernel_size - 1) + 1 - stride) / 2.))
        else:
            self.padding = padding
        self.dilation = dilation
        assert type(groups) == int
        if kernel_size == 1:
            self.groups = 1
        else:
            self.groups = groups
        self.bias = bias
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)

        if slimmable:
            self.conv = nn.Sequential(
                USConv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias, width_mult_list=width_mult_list),
                USBatchNorm2d(C_out, width_mult_list),
                nn.ReLU(inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size, stride, padding=self.padding, dilation=dilation, groups=self.groups, bias=bias),
                # nn.BatchNorm2d(C_out),
                BatchNorm2d(C_out, eps=1e-03, affine=True),
                nn.ReLU(inplace=True),
            )
    
    def set_ratio(self, ratio):
        assert self.slimmable
        assert len(ratio) == 2
        self.ratio = ratio
        self.conv[0].set_ratio(ratio)
        self.conv[1].set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops
    
    @staticmethod
    def _latency(h, w, C_in, C_out, kernel_size=3, stride=1, padding=None, dilation=1, groups=1, bias=False):
        layer = ConvNorm(C_in, C_out, kernel_size, stride, padding, dilation, groups, bias, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0]), "c_in %d, self.C_in * self.ratio[0] %d"%(c_in, self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in, "c_in %d, self.C_in %d"%(c_in, self.C_in)
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "ConvNorm_H%d_W%d_Cin%d_Cout%d_kernel%d_stride%d"%(h_in, w_in, c_in, c_out, self.kernel_size, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = ConvNorm._latency(h_in, w_in, c_in, c_out, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        assert x.size()[1] == self.C_in, "{} {}".format(x.size()[1], self.C_in)
        x = self.conv(x)
        return x


class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, stride=1, slimmable=True, width_mult_list=[1.]):
        super(FactorizedReduce, self).__init__()
        assert stride in [1, 2]
        assert C_out % 2 == 0
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        self.slimmable = slimmable
        self.width_mult_list = width_mult_list
        self.ratio = (1., 1.)
        if stride == 1 and slimmable:
            self.conv1 = USConv2d(C_in, C_out, 1, stride=1, padding=0, bias=False, width_mult_list=width_mult_list)
            self.bn = USBatchNorm2d(C_out, width_mult_list)
            self.relu = nn.ReLU(inplace=True)
        elif stride == 2:
            self.relu = nn.ReLU(inplace=True)
            if slimmable:
                self.conv1 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.conv2 = USConv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False, width_mult_list=width_mult_list)
                self.bn = USBatchNorm2d(C_out, width_mult_list)
            else:
                self.conv1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
                self.conv2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
                self.bn = BatchNorm2d(C_out)

    def set_ratio(self, ratio):
        assert len(ratio) == 2
        if self.stride == 1:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])
        elif self.stride == 2:
            self.ratio = ratio
            self.conv1.set_ratio(ratio)
            self.conv2.set_ratio(ratio)
            self.bn.set_ratio(ratio[1])

    @staticmethod
    def _flops(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        flops, params = profile(layer, inputs=(torch.randn(1, C_in, h, w),), verbose=False)
        return flops

    @staticmethod
    def _latency(h, w, C_in, C_out, stride=1):
        layer = FactorizedReduce(C_in, C_out, stride, slimmable=False)
        latency = compute_latency(layer, (1, C_in, h, w))
        return latency

    def forward_latency(self, size):
        c_in, h_in, w_in = size
        if self.slimmable:
            assert c_in == int(self.C_in * self.ratio[0])
            c_out = int(self.C_out * self.ratio[1])
        else:
            assert c_in == self.C_in
            c_out = self.C_out
        if self.stride == 1:
            h_out = h_in; w_out = w_in
        else:
            h_out = h_in // 2; w_out = w_in // 2
        name = "FactorizedReduce_H%d_W%d_Cin%d_Cout%d_stride%d"%(h_in, w_in, c_in, c_out, self.stride)
        if name in latency_lookup_table:
            latency = latency_lookup_table[name]
        else:
            print("not found in latency_lookup_table:", name)
            latency = FactorizedReduce._latency(h_in, w_in, c_in, c_out, self.stride)
            latency_lookup_table[name] = latency
            np.save(table_file_name, latency_lookup_table)
        return latency, (c_out, h_out, w_out)

    def forward(self, x):
        if self.stride == 2:
            out = torch.cat([self.conv1(x), self.conv2(x[:,:,1:,1:])], dim=1)
            out = self.bn(out)
            out = self.relu(out)
            return out
        else:
            if self.slimmable:
                out = self.conv1(x)
                out = self.bn(out)
                out = self.relu(out)
                return out
            else:
                return x


from collections import OrderedDict
# OPS = {
#     'skip': lambda C_in, dropprob, dilated=1: FactorizedReduce(C_in, C_in, stride=1, slimmable=False, width_mult_list=[1.]),
#     'non_bottleneck_1d_1': lambda C_in, dropprob, dilated=1: non_bottleneck_1d(C_in, dropprob, dilated),
#     'non_bottleneck_1d_2': lambda C_in, dropprob, dilated=2: non_bottleneck_1d(C_in, dropprob, dilated),
#     'non_bottleneck_1d_4': lambda C_in, dropprob, dilated=4: non_bottleneck_1d(C_in, dropprob, dilated),
#     'non_bottleneck_1d_8': lambda C_in, dropprob, dilated=8: non_bottleneck_1d(C_in, dropprob, dilated),
#     'non_bottleneck_1d_16': lambda C_in, dropprob, dilated=16: non_bottleneck_1d(C_in, dropprob, dilated),
# }

OPS = {
    'skip': lambda C_in, dropprob: FactorizedReduce(C_in, C_in, stride=1, slimmable=False, width_mult_list=[1.]),
    'non_bottleneck_1d_r1': lambda C_in, C_out, stride, dropprob, dlt=1: non_bottleneck_1d_rn(C_in, C_out, stride, dropprob, dlt),
    'non_bottleneck_1d_r2': lambda C_in, C_out, stride, dropprob, dlt=2: non_bottleneck_1d_rn(C_in, C_out, stride, dropprob, dlt),
    'non_bottleneck_1d_r3': lambda C_in, C_out, stride, dropprob, dlt=4: non_bottleneck_1d_rn(C_in, C_out, stride, dropprob, dlt),
    'non_bottleneck_1d_r4': lambda C_in, C_out, stride, dropprob, dlt=8: non_bottleneck_1d_rn(C_in, C_out, stride, dropprob, dlt),
    'non_bottleneck_1d_r5': lambda C_in, C_out, stride, dropprob, dlt=16: non_bottleneck_1d_rn(C_in, C_out, stride, dropprob, dlt),
}