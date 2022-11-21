import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from thop import profile
from genotypes2 import CHANNELS


class non_bottleneck_1d_rn_pool(nn.Module):
    def __init__(self, C_in, C_out, dropprob, dlt, pool):
        super().__init__()
        self.conv3x1 = nn.Conv2d(C_in, C_out, (3, 1), stride=1, padding=(dlt, 0), bias=True, dilation=(dlt, 1))
        self.bn0 = nn.BatchNorm2d(C_out, eps=1e-03)
        self.conv1x3 = nn.Conv2d(C_out, C_out, (1, 3), stride=1, padding=(0, dlt), bias=True, dilation=(1, dlt))
        self.bn1 = nn.BatchNorm2d(C_out, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.pool = nn.MaxPool2d(pool, stride=pool)
        self._pool = pool

    def forward(self, input, chan_mask):
        output = self.pool(input)
        output = self.bn0(self.conv3x1(output))
        output = chan_mask * output
        output = F.relu(output)
        output = self.bn1(self.conv1x3(output))
        output = F.interpolate(output, size=(int(input.size(2)), int(input.size(3))), mode='bilinear', align_corners=True)
        if (self.dropout.p != 0):
            output = self.dropout(output)
        return output + input

    def forward_flops(self, in_ch, in_height, in_width):
        base_flops = ((3*1*in_ch[0] + 1)*in_ch[1] + (1*3*in_ch[1] + 1) * in_ch[2]) + self._pool*self._pool*in_ch[0]
        return base_flops*int(in_height/self._pool)*int(in_width/self._pool)


class non_bottleneck_1d_rn(nn.Module):
    def __init__(self, C_in, C_out, dropprob, dlt):
        super().__init__()
        self.conv3x1 = nn.Conv2d(C_in, C_out, (3, 1), stride=1, padding=(dlt, 0), bias=True, dilation=(dlt, 1))
        self.bn0 = nn.BatchNorm2d(C_out, eps=1e-03)
        self.conv1x3 = nn.Conv2d(C_out, C_out, (1, 3), stride=1, padding=(0, dlt), bias=True, dilation=(1, dlt))
        self.bn1 = nn.BatchNorm2d(C_out, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input, chan_mask):
        # output = F.relu(self.conv3x1(input))
        output = self.bn0(self.conv3x1(input))
        output = chan_mask * output
        output = F.relu(output)
        output = self.bn1(self.conv1x3(output))
        if (self.dropout.p != 0):
            output = self.dropout(output)
        return output + input

    def forward_flops(self, in_ch, in_height, in_width):
        base_flops = ((3*1*in_ch[0] + 1)*in_ch[1] + (1*3*in_ch[1] + 1) * in_ch[2])
        return base_flops*in_height*in_width


class non_bottleneck_1d_rn_re(nn.Module):
    def __init__(self, prun_chan, dropprob, dlt):
        super().__init__()
        self.conv3x1 = nn.Conv2d(prun_chan[0], prun_chan[1], (3, 1), stride=1, padding=(dlt, 0), bias=True, dilation=(dlt, 1))
        self.bn0 = nn.BatchNorm2d(prun_chan[1], eps=1e-03)
        self.conv1x3 = nn.Conv2d(prun_chan[1], prun_chan[2], (1, 3), stride=1, padding=(0, dlt), bias=True, dilation=(1, dlt))
        self.bn1 = nn.BatchNorm2d(prun_chan[2], eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        # output = F.relu(self.conv3x1(input))
        output = self.bn0(self.conv3x1(input))
        output = F.relu(output)
        output = self.bn1(self.conv1x3(output))
        if (self.dropout.p != 0):
            output = self.dropout(output)

        input_chan = input.size(1)
        output_chan = output.size(1)
        if input_chan >= output_chan:
            return output + input[:, :output_chan :, :]
        else:
            return torch.cat([input+output[:, :input_chan, :, :], output[:, input_chan:, :, :]], dim=1)


class non_bottleneck_1d_rn_re_pool(nn.Module):
    def __init__(self, prun_chan, dropprob, dlt, pool):
        super().__init__()
        self.conv3x1 = nn.Conv2d(prun_chan[0], prun_chan[1], (3, 1), stride=1, padding=(dlt, 0), bias=True, dilation=(dlt, 1))
        self.bn0 = nn.BatchNorm2d(prun_chan[1], eps=1e-03)
        self.conv1x3 = nn.Conv2d(prun_chan[1], prun_chan[2], (1, 3), stride=1, padding=(0, dlt), bias=True, dilation=(1, dlt))
        self.bn1 = nn.BatchNorm2d(prun_chan[2], eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)
        self.pool = nn.MaxPool2d(pool, stride=pool)


    def forward(self, input):
        output = self.pool(input)
        output = self.bn0(self.conv3x1(output))
        output = F.relu(output)
        output = self.bn1(self.conv1x3(output))
        output = F.interpolate(output, size=(int(input.size(2)), int(input.size(3))), mode='bilinear', align_corners=True)
        if (self.dropout.p != 0):
            output = self.dropout(output)

        input_chan = input.size(1)
        output_chan = output.size(1)
        if input_chan >= output_chan:
            return output + input[:, :output_chan:, :]
        else:
            return torch.cat([input+output[:, :input_chan, :, :], output[:, input_chan:, :, :]], dim=1)



# class non_bottleneck_1d_r1(nn.Module):
#     def __init__(self, chann, dropprob):
#         super().__init__()
#
#         self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
#
#         self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
#
#         self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.dropout = nn.Dropout2d(dropprob)
#
#     def forward(self, input):
#         output = self.conv3x1_1(input)
#         output = F.relu(output)
#         output = self.conv1x3_1(output)
#         output = self.bn1(output)
#
#         if (self.dropout.p != 0):
#             output = self.dropout(output)
#
#         # return F.relu(output + input)  # +input = identity (residual connection)
#         return output + input   # +input = identity (residual connection)
#
# class non_bottleneck_1d_r2(nn.Module):
#     def __init__(self, chann, dropprob):
#         super().__init__()
#
#         # self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
#         #
#         # self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
#         #
#         # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 2, 0), bias=True,
#                                    dilation=(2, 1))
#
#         self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 2), bias=True,
#                                    dilation=(1, 2))
#
#         self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.dropout = nn.Dropout2d(dropprob)
#
#     def forward(self, input):
#         # output1 = self.conv3x1_1(input)
#         # output1 = F.relu(output1)
#         # output1 = self.conv1x3_1(output1)
#         # output1 = self.bn1(output1)
#         # # output1 = F.relu(output1)
#
#         output = self.conv3x1_2(input)
#         output = F.relu(output)
#         output = self.conv1x3_2(output)
#         output = self.bn2(output)
#
#         # output = output1+output2
#         if (self.dropout.p != 0):
#             output = self.dropout(output)
#
#         # return F.relu(output + input)  # +input = identity (residual connection)
#         return output + input   # +input = identity (residual connection)
#
# class non_bottleneck_1d_r3(nn.Module):
#     def __init__(self, chann, dropprob):
#         super().__init__()
#
#         # self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
#         #
#         # self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
#         #
#         # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
#         #
#         # self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 2, 0), bias=True,
#         #                            dilation=(2, 1))
#         #
#         # self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 2), bias=True,
#         #                            dilation=(1, 2))
#         #
#         # self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.conv3x1_3 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 4, 0), bias=True,
#                                    dilation=(4, 1))
#
#         self.conv1x3_3 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 4), bias=True,
#                                    dilation=(1, 4))
#
#         self.bn3 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.dropout = nn.Dropout2d(dropprob)
#
#     def forward(self, input):
#         # output1 = self.conv3x1_1(input)
#         # output1 = F.relu(output1)
#         # output1 = self.conv1x3_1(output1)
#         # output1 = self.bn1(output1)
#         # # output1 = F.relu(output1)
#         #
#         # output2 = self.conv3x1_2(input)
#         # output2 = F.relu(output2)
#         # output2 = self.conv1x3_2(output2)
#         # output2 = self.bn2(output2)
#         # # output2 = F.relu(output2)
#
#         output = self.conv3x1_3(input)
#         output = F.relu(output)
#         output = self.conv1x3_3(output)
#         output = self.bn3(output)
#
#         # output = output1+output2+output3
#         if (self.dropout.p != 0):
#             output = self.dropout(output)
#
#         # return F.relu(output + input)  # +input = identity (residual connection)
#         return output + input   # +input = identity (residual connection)
#
# class non_bottleneck_1d_r4(nn.Module):
#     def __init__(self, chann, dropprob):
#         super().__init__()
#
#         # self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
#         #
#         # self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
#         #
#         # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
#         #
#         # self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 2, 0), bias=True,
#         #                            dilation=(2, 1))
#         #
#         # self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 2), bias=True,
#         #                            dilation=(1, 2))
#         #
#         # self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
#         #
#         # self.conv3x1_3 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 4, 0), bias=True,
#         #                            dilation=(4, 1))
#         #
#         # self.conv1x3_3 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 4), bias=True,
#         #                            dilation=(1, 4))
#         #
#         # self.bn3 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.conv3x1_4 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 8, 0), bias=True,
#                                    dilation=(8, 1))
#
#         self.conv1x3_4 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 8), bias=True,
#                                    dilation=(1, 8))
#
#         self.bn4 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.dropout = nn.Dropout2d(dropprob)
#
#     def forward(self, input):
#         # output1 = self.conv3x1_1(input)
#         # output1 = F.relu(output1)
#         # output1 = self.conv1x3_1(output1)
#         # output1 = self.bn1(output1)
#         # # output1 = F.relu(output1)
#         #
#         # output2 = self.conv3x1_2(input)
#         # output2 = F.relu(output2)
#         # output2 = self.conv1x3_2(output2)
#         # output2 = self.bn2(output2)
#         # # output2 = F.relu(output2)
#         #
#         # output3 = self.conv3x1_3(input)
#         # output3 = F.relu(output3)
#         # output3 = self.conv1x3_3(output3)
#         # output3 = self.bn3(output3)
#         # # output3 = F.relu(output3)
#
#         output = self.conv3x1_4(input)
#         output = F.relu(output)
#         output = self.conv1x3_4(output)
#         output = self.bn4(output)
#
#         # output = output1+output2+output3+output4
#         if (self.dropout.p != 0):
#             output = self.dropout(output)
#
#         # return F.relu(output + input)  # +input = identity (residual connection)
#         return output + input   # +input = identity (residual connection)
#
# class non_bottleneck_1d_r5(nn.Module):
#     def __init__(self, chann, dropprob):
#         super().__init__()
#
#         # self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
#         #
#         # self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
#         #
#         # self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
#         #
#         # self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 2, 0), bias=True,
#         #                            dilation=(2, 1))
#         #
#         # self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 2), bias=True,
#         #                            dilation=(1, 2))
#         #
#         # self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
#         #
#         # self.conv3x1_3 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 4, 0), bias=True,
#         #                            dilation=(4, 1))
#         #
#         # self.conv1x3_3 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 4), bias=True,
#         #                            dilation=(1, 4))
#         #
#         # self.bn3 = nn.BatchNorm2d(chann, eps=1e-03)
#         #
#         # self.conv3x1_4 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 8, 0), bias=True,
#         #                            dilation=(8, 1))
#         #
#         # self.conv1x3_4 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 8), bias=True,
#         #                            dilation=(1, 8))
#         #
#         # self.bn4 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.conv3x1_5 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * 16, 0), bias=True,
#                                    dilation=(16, 1))
#
#         self.conv1x3_5 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * 16), bias=True,
#                                    dilation=(1, 16))
#
#         self.bn5 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.dropout = nn.Dropout2d(dropprob)
#
#     def forward(self, input):
#         # output1 = self.conv3x1_1(input)
#         # output1 = F.relu(output1)
#         # output1 = self.conv1x3_1(output1)
#         # output1 = self.bn1(output1)
#         # # output1 = F.relu(output1)
#         #
#         # output2 = self.conv3x1_2(input)
#         # output2 = F.relu(output2)
#         # output2 = self.conv1x3_2(output2)
#         # output2 = self.bn2(output2)
#         # # output2 = F.relu(output2)
#         #
#         # output3 = self.conv3x1_3(input)
#         # output3 = F.relu(output3)
#         # output3 = self.conv1x3_3(output3)
#         # output3 = self.bn3(output3)
#         # # output3 = F.relu(output3)
#         #
#         # output4 = self.conv3x1_4(input)
#         # output4 = F.relu(output4)
#         # output4 = self.conv1x3_4(output4)
#         # output4 = self.bn4(output4)
#         # # output4 = F.relu(output4)
#
#         output = self.conv3x1_5(input)
#         output = F.relu(output)
#         output = self.conv1x3_5(output)
#         output = self.bn5(output)
#
#         # output = output1+output2+output3+output4+output5
#         if (self.dropout.p != 0):
#             output = self.dropout(output)
#
#         # return F.relu(output + input)  # +input = identity (residual connection)
#         return output + input  # +input = identity (residual connection)
#
# class non_bottleneck_1d(nn.Module):
#     def __init__(self, chann, dropprob, dilated):
#         super().__init__()
#
#         self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)
#
#         self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)
#
#         self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1 * dilated, 0), bias=True,
#                                    dilation=(dilated, 1))
#
#         self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1 * dilated), bias=True,
#                                    dilation=(1, dilated))
#
#         self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
#
#         self.dropout = nn.Dropout2d(dropprob)
#
#     def forward(self, input):
#         output = self.conv3x1_1(input)
#         output = F.relu(output)
#         output = self.conv1x3_1(output)
#         output = self.bn1(output)
#         output = F.relu(output)
#
#         output = self.conv3x1_2(output)
#         output = F.relu(output)
#         output = self.conv1x3_2(output)
#         output = self.bn2(output)
#
#         if (self.dropout.p != 0):
#             output = self.dropout(output)
#
#         # return F.relu(output + input)  # +input = identity (residual connection)
#         return output + input  # +input = identity (residual connection)


# OPS = {
#     'non_bottleneck_1d_r1': lambda C_in, dropprob: non_bottleneck_1d_r1(C_in, dropprob),
#     'non_bottleneck_1d_r2': lambda C_in, dropprob: non_bottleneck_1d_r2(C_in, dropprob),
#     'non_bottleneck_1d_r3': lambda C_in, dropprob: non_bottleneck_1d_r3(C_in, dropprob),
#     'non_bottleneck_1d_r4': lambda C_in, dropprob: non_bottleneck_1d_r4(C_in, dropprob),
#     'non_bottleneck_1d_r5': lambda C_in, dropprob: non_bottleneck_1d_r5(C_in, dropprob),
# }

OPS = {
    'non_bottleneck_1d_r1': lambda C_in, dropprob, dlt=1: non_bottleneck_1d_rn(C_in, C_in, dropprob, dlt),
    'non_bottleneck_1d_r2': lambda C_in, dropprob, dlt=2: non_bottleneck_1d_rn(C_in, C_in, dropprob, dlt),
    'non_bottleneck_1d_r3': lambda C_in, dropprob, dlt=4: non_bottleneck_1d_rn(C_in, C_in, dropprob, dlt),
    'non_bottleneck_1d_r4': lambda C_in, dropprob, dlt=8: non_bottleneck_1d_rn(C_in, C_in, dropprob, dlt),
    'non_bottleneck_1d_r5': lambda C_in, dropprob, dlt=16: non_bottleneck_1d_rn(C_in, C_in, dropprob, dlt),
    'non_bottleneck_1d_r1_p2': lambda C_in, dropprob, dlt=1, pool=2: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    'non_bottleneck_1d_r2_p2': lambda C_in, dropprob, dlt=2, pool=2: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    'non_bottleneck_1d_r3_p2': lambda C_in, dropprob, dlt=4, pool=2: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    'non_bottleneck_1d_r4_p2': lambda C_in, dropprob, dlt=8, pool=2: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    'non_bottleneck_1d_r5_p2': lambda C_in, dropprob, dlt=16, pool=2: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r1_p3': lambda C_in, dropprob, dlt=1, pool=3: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r2_p3': lambda C_in, dropprob, dlt=2, pool=3: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r3_p3': lambda C_in, dropprob, dlt=4, pool=3: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r4_p3': lambda C_in, dropprob, dlt=8, pool=3: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r5_p3': lambda C_in, dropprob, dlt=16, pool=3: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r1_p4': lambda C_in, dropprob, dlt=1, pool=4: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r2_p4': lambda C_in, dropprob, dlt=2, pool=4: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r3_p4': lambda C_in, dropprob, dlt=4, pool=4: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r4_p4': lambda C_in, dropprob, dlt=8, pool=4: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
    # 'non_bottleneck_1d_r5_p4': lambda C_in, dropprob, dlt=16, pool=4: non_bottleneck_1d_rn_pool(C_in, C_in, dropprob, dlt, pool),
}

OPS_RE = {
    'non_bottleneck_1d_r1': lambda prun_chan, dropprob, dlt=1: non_bottleneck_1d_rn_re(prun_chan, dropprob, dlt),
    'non_bottleneck_1d_r2': lambda prun_chan, dropprob, dlt=2: non_bottleneck_1d_rn_re(prun_chan, dropprob, dlt),
    'non_bottleneck_1d_r3': lambda prun_chan, dropprob, dlt=4: non_bottleneck_1d_rn_re(prun_chan, dropprob, dlt),
    'non_bottleneck_1d_r4': lambda prun_chan, dropprob, dlt=8: non_bottleneck_1d_rn_re(prun_chan, dropprob, dlt),
    'non_bottleneck_1d_r5': lambda prun_chan, dropprob, dlt=16: non_bottleneck_1d_rn_re(prun_chan, dropprob, dlt),
    'non_bottleneck_1d_r1_p2': lambda prun_chan, dropprob, dlt=1, pool=2: non_bottleneck_1d_rn_re_pool(prun_chan, dropprob, dlt, pool),
    'non_bottleneck_1d_r2_p2': lambda prun_chan, dropprob, dlt=2, pool=2: non_bottleneck_1d_rn_re_pool(prun_chan, dropprob, dlt, pool),
    'non_bottleneck_1d_r3_p2': lambda prun_chan, dropprob, dlt=4, pool=2: non_bottleneck_1d_rn_re_pool(prun_chan, dropprob, dlt, pool),
    'non_bottleneck_1d_r4_p2': lambda prun_chan, dropprob, dlt=8, pool=2: non_bottleneck_1d_rn_re_pool(prun_chan, dropprob, dlt, pool),
    'non_bottleneck_1d_r5_p2': lambda prun_chan, dropprob, dlt=16, pool=2: non_bottleneck_1d_rn_re_pool(prun_chan, dropprob, dlt, pool),
}
