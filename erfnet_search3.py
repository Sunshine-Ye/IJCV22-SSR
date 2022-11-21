# ERFNet full model definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from genotypes2 import PRIMITIVES, CHANNELS
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

    def forward_flops(self, nin, nout, out_height, out_width):
        base_flops = (3 * 3 * nin + 1) * (nout-nin) + 2 * 2 * nin
        return base_flops*out_height*out_width
    

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):        
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated, 0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))

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
        
        return F.relu(output+input)    #+input = identity (residual connection)


class Encoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        for x in range(0, 5):    # 5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):    #2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        #Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Decoder (nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlock(128,64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlock(64,16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# ERFNet
class Net(nn.Module):
    def __init__(self, num_classes, encoder=None):  # use encoder to pass pretrained encoder
        super().__init__()

        if (encoder == None):
            self.encoder = Encoder(num_classes)
        else:
            self.encoder = encoder
        self.decoder = Decoder(num_classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)    # predict=False by default
            return self.decoder.forward(output)


class MixedOp(nn.Module):
  def __init__(self, C_in, dropprob, switch_cell):
    super(MixedOp, self).__init__()
    self.mixed_dw_ops = nn.ModuleList()
    for i in range(len(switch_cell)):
        if switch_cell[i]:
            primitive = PRIMITIVES[i]
            op = OPS[primitive](C_in, dropprob)
            self.mixed_dw_ops.append(op)
            del op
    # create masking for each op
    self.masking = torch.ones(len(CHANNELS), 1, C_in, 1, 1).cuda()
    for j in range(len(CHANNELS)):
        # chan_keep = int(CHANNELS[j] * C_in)
        chan_keep = int(C_in-CHANNELS[j])
        self.masking[j][0][chan_keep:] = 0

  def forward(self, x, weights_o, weights_oc, weights_occ, in_ch):
    ## 15578M(0.2230)
    # out1 = sum(w_o * sum(w_c*mask for w_c, mask in zip(w_oc, self.masking) if w_c != 0) *
    #           op(x, sum(w_cc*mask for w_cc, mask in zip(w_occ, self.masking) if w_cc != 0))
    #           for w_o, w_oc, w_occ, op in zip(weights_o, weights_oc, weights_occ, self.mixed_dw_ops) if w_o != 0)
    ## 15626M(0.2155)  (66.51)
    out = 0
    expect_flops = 0
    (in_height, in_width) = x.size()[2:]
    for w_o, w_oc, w_occ, op in zip(weights_o, weights_oc, weights_occ, self.mixed_dw_ops):
        if w_o != 0:
            mid_mask = sum(w_cc*mask for w_cc, mask in zip(w_occ, self.masking) if w_cc != 0)
            lst_mask = sum(w_c*mask for w_c, mask in zip(w_oc, self.masking) if w_c != 0)
            out += w_o * lst_mask * op(x, mid_mask)
            mid_ch = torch.sum(mid_mask)
            lst_ch = torch.sum(lst_mask)
            expect_flops += w_o * op.forward_flops([in_ch, mid_ch, lst_ch], in_height, in_width)
    ## 15626M(0.1725)  (65.80)
    # out = 0
    # for w_o, w_oc, w_occ, op in zip(weights_o, weights_oc, weights_occ, self.mixed_dw_ops):
    #     if w_o != 0:
    #         mid_mask = torch.sum(w_occ.view(-1, 1, 1, 1, 1) * self.masking, 0)
    #         lst_mask = torch.sum(w_oc.view(-1, 1, 1, 1, 1) * self.masking, 0)
    #         out += w_o * lst_mask * op(x, mid_mask)
    return out, expect_flops, lst_ch


class Encoder_search(nn.Module):
    def __init__(self, num_classes, switch_cell, switch_cell_c, switch_cell_cc):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        cnt = 0

        self.layers.append(DownsamplerBlock(16, 64))  # (16, 64)
        for x in range(0, 5):    #5 times
           self.layers.append(MixedOp(64, 0.03, switch_cell[cnt]))
           cnt += 1

        self.layers.append(DownsamplerBlock(64, 128))  # (64, 128) (96, 160)
        for x in range(0, 8):    #2 times
            self.layers.append(MixedOp(128, 0.3, switch_cell[cnt]))
            cnt += 1

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)
        self.switch_cell = switch_cell
        self.switch_cell_c = switch_cell_c
        self.switch_cell_cc = switch_cell_cc
        self._initialize_alphas()

    def forward(self, input, predict=True):
        weights_o = torch.sigmoid(self.alphas)
        valid_weights_o = torch.where(torch.tensor(self.switch_cell).cuda(), weights_o, torch.zeros(13, 10).cuda())
        norm_weights_o = torch.div(valid_weights_o, torch.sum(valid_weights_o, 1, keepdim=True))

        weights_oc = torch.sigmoid(self.alphas1)
        valid_weights_oc = torch.where(torch.tensor(self.switch_cell_c, dtype=torch.uint8).cuda(), weights_oc, torch.zeros(13, 10, 9).cuda())
        norm_weights_oc = torch.div(valid_weights_oc, torch.sum(valid_weights_oc, 2, keepdim=True)).unsqueeze(-1)

        weights_occ = torch.sigmoid(self.alphas2)
        valid_weights_occ = torch.where(torch.tensor(self.switch_cell_cc, dtype=torch.uint8).cuda(), weights_occ, torch.zeros(13, 10, 9).cuda())
        norm_weights_occ = torch.div(valid_weights_occ, torch.sum(valid_weights_occ, 2, keepdim=True)).unsqueeze(-1)

        output = self.initial_block(input)
        total_flops = self.initial_block.forward_flops(3, 16, output.size(2), output.size(3))
        cnt = 0
        for index, layer in enumerate(self.layers):
            if index == 0 or index == 6:
                output = layer(output)
                if index == 0:
                    total_flops += layer.forward_flops(16, 64, output.size(2), output.size(3))
                    in_ch = 64
                else:
                    total_flops += layer.forward_flops(in_ch, 128, output.size(2), output.size(3))
                    in_ch = 128
            else:
                output, expect_flops, lst_ch = layer(output, norm_weights_o[cnt], norm_weights_oc[cnt], norm_weights_occ[cnt], in_ch)
                output = F.relu(output)
                cnt += 1
                total_flops += expect_flops
                in_ch = lst_ch

        if predict:
            output = self.output_conv(output)
            total_flops += (1*1*128 + 1)*20*output.size(2)*output.size(3)

        return output, total_flops / 1e9  # 1G=1e9 1M=1e6

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)
        num_chs = len(CHANNELS)
        self.alphas = Variable(torch.zeros(len(self.layers)-2, num_ops).cuda(), requires_grad=True)
        self.alphas1 = Variable(torch.zeros(len(self.layers) - 2, num_ops, num_chs).cuda(), requires_grad=True)
        self.alphas2 = Variable(torch.zeros(len(self.layers) - 2, num_ops, num_chs).cuda(), requires_grad=True)
        self._arch_parameters = [self.alphas]+[self.alphas1]+[self.alphas2]

    def arch_parameters(self):
        return self._arch_parameters

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()]
        # return [param for name, param in self.named_parameters() if name not in ['alphas', 'alphas1']]


if __name__ == '__main__' :
    switch_cell = []
    for i in range(13):
        switch_cell.append([True, True, True, True, True, True])
    model = Encoder_search(20, switch_cell)
    x = torch.tensor(torch.ones(4, 3, 224, 224))
    y = model(x, predict=True)
    print(model.weight_parameters())
    print(y.shape)