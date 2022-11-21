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
from operations4 import *


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

        self.layers.append(DownsamplerBlock(16,64))

        for x in range(0, 5):    #5 times
           self.layers.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers.append(DownsamplerBlock(64,128))

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
  def __init__(self, C_in, C_out, stride, dropprob, switch_cell):
    super(MixedOp, self).__init__()
    self.mixed_dw_ops = nn.ModuleList()
    for i in range(len(switch_cell)):
        if switch_cell[i]:
            primitive = PRIMITIVES[i]
            # op = OPS[primitive](C, stride, False)
            op = OPS[primitive](C_in, C_out, stride, dropprob)
            self.mixed_dw_ops.append(op)
            del op

  def forward(self, x, weights_o):
    out = sum(w_o * op(x) for w_o, op in zip(weights_o,  self.mixed_dw_ops))
    return out


class Encoder_search(nn.Module):
    def __init__(self, num_classes, switch_cell):
        super().__init__()
        self.initial_block = DownsamplerBlock(3, 16)
        self.layers = nn.ModuleList()
        cnt = 0

        self.layers.append(MixedOp(16, 64, 2, 0.0, switch_cell[cnt]))
        cnt += 1
        for x in range(0, 5):    #5 times
           self.layers.append(MixedOp(64, 64, 1, 0.03, switch_cell[cnt]))
           cnt += 1

        self.layers.append(MixedOp(64, 128, 2, 0.0, switch_cell[cnt]))
        cnt += 1
        for x in range(0, 8):    #2 times
            self.layers.append(MixedOp(128, 128, 1, 0.3, switch_cell[cnt]))
            cnt += 1

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)
        self._initialize_alphas()

    def forward(self, input, predict=True):

        weights_o = torch.sigmoid(self.alphas)
        weights_r = torch.sum(weights_o, 1, keepdim=True)
        weights_o = torch.div(weights_o, weights_r)
        # print(weights_o)
        # one = torch.ones_like(weights_o)
        # zero = torch.zeros_like(weights_o)
        # weights_o = torch.where(weights_o > 0.9, one, weights_o)
        # weights_o = torch.where(weights_o < 0.1, zero, weights_o)

        output = self.initial_block(input)

        for index, layer in enumerate(self.layers):
            output = layer(output, weights_o[index])
            output = F.relu(output)
            print(output.shape)

        if predict:
            output = self.output_conv(output)

        return output

    def _initialize_alphas(self):
        num_ops = len(PRIMITIVES)
        self.alphas = Variable(torch.zeros(len(self.layers), num_ops).cuda(), requires_grad=True)
        print(self.alphas.shape)

        self._arch_parameters = [self.alphas]

    def arch_parameters(self):
        return self._arch_parameters

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()]



if __name__ == '__main__' :
    switch_cell = []
    for i in range(13):
        switch_cell.append([True, True, True, True, True, True])
    model = Encoder_search(20, switch_cell)
    x = torch.tensor(torch.ones(4, 3, 224, 224))
    y = model(x, predict=True)
    print(model.weight_parameters())
    print(y.shape)