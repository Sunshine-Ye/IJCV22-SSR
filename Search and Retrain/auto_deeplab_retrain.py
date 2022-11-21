import torch
import torch.nn as nn
import numpy as np
import cell_level_retrain
from genotypes1 import PRIMITIVES
import torch.nn.functional as F
from torch.autograd import Variable
from operations1 import *
from torchstat import stat


def parse_init(switch_net, filter_multiplier_cur, times=[0.5, 1, 2]):
    prev_fmultipliers = []
    for j in range(len(switch_net)):
        if switch_net[j]:
            prev_fmultipliers.append(int(filter_multiplier_cur*times[j]))
        else:
            prev_fmultipliers.append(None)
    return prev_fmultipliers


def parse_forward(switch_net, possible_inputs=[]):
    prev_features = []
    for j in range(len(switch_net)):
        if switch_net[j]:
            prev_features.append(possible_inputs[j])
        else:
            prev_features.append(None)
    return prev_features


class AutoDeeplab (nn.Module):
    def __init__(self, num_classes, num_layers, filter_multiplier=8, group_multiplier=8,
                 criterion=None,  switches_cell=[], switches_net=[], cell=cell_level_retrain.Cell):
        super(AutoDeeplab, self).__init__()

        self._num_classes = num_classes
        self._num_layers = num_layers
        self._filter_multiplier = filter_multiplier
        self._group_multiplier = group_multiplier
        self._criterion = criterion
        self._switches_net = switches_net

        f_initial = int(self._filter_multiplier)
        half_f_initial = int(f_initial / 2)

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, half_f_initial * self._group_multiplier, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(half_f_initial * self._group_multiplier, eps=1e-03, affine=True),
            nn.ReLU(inplace=True)
        )
        self.stem1 = nn.Sequential(
            nn.Conv2d(half_f_initial * self._group_multiplier, half_f_initial * self._group_multiplier, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(half_f_initial * self._group_multiplier, eps=1e-03, affine=True),
            nn.ReLU(inplace=True)
        )
        self.stem2 = nn.Sequential(
            nn.Conv2d(half_f_initial * self._group_multiplier, f_initial * self._group_multiplier, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(f_initial * self._group_multiplier, eps=1e-03, affine=True),
            nn.ReLU(inplace=True)
        )

        # intitial_fm = C_initial
        self.cells = nn.ModuleList()
        for i in range(self._num_layers):
            if i == 0:
                if switches_net[i][0] == [False, False, False]:
                    cell1 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][0], self._filter_multiplier, times=[0.5, 1, 2])
                    cell1 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier, switches_cell[i][0])
                if switches_net[i][1] == [False, False, False]:
                    cell2 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][1], self._filter_multiplier * 2, times=[0.5, 1, 2])
                    cell2 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier * 2, switches_cell[i][1])
                self.cells += [cell1]
                self.cells += [cell2]
                del cell1, cell2
            elif i == 1:
                if switches_net[i][0] == [False, False, False]:
                    cell1 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][0], self._filter_multiplier, times=[0.5, 1, 2])
                    cell1 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier, switches_cell[i][0])
                if switches_net[i][1] == [False, False, False]:
                    cell2 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][1], self._filter_multiplier * 2, times=[0.5, 1, 2])
                    cell2 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier * 2, switches_cell[i][1])
                if switches_net[i][2] == [False, False, False]:
                    cell3 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][2], self._filter_multiplier * 4, times=[0.5, 1, 2])
                    cell3 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier * 4, switches_cell[i][2])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                del cell1, cell2, cell3
            elif i == 2:
                if switches_net[i][0] == [False, False, False]:
                    cell1 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][0], self._filter_multiplier, times=[0.5, 1, 2])
                    cell1 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier, switches_cell[i][0])
                if switches_net[i][1] == [False, False, False]:
                    cell2 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][1], self._filter_multiplier * 2, times=[0.5, 1, 2])
                    cell2 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier * 2, switches_cell[i][1])
                if switches_net[i][2] == [False, False, False]:
                    cell3 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][2], self._filter_multiplier * 4, times=[0.5, 1, 2])
                    cell3 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier * 4, switches_cell[i][2])
                if switches_net[i][3] == [False, False, False]:
                    cell4 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][3], self._filter_multiplier * 8, times=[0.5, 1, 2])
                    cell4 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier * 8, switches_cell[i][3])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]
                del cell1, cell2, cell3, cell4
            else:
                if switches_net[i][0] == [False, False, False]:
                    cell1 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][0], self._filter_multiplier, times=[0.5, 1, 2])
                    cell1 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier, switches_cell[i][0])
                if switches_net[i][1] == [False, False, False]:
                    cell2 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][1], self._filter_multiplier * 2, times=[0.5, 1, 2])
                    cell2 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier * 2, switches_cell[i][1])
                if switches_net[i][2] == [False, False, False]:
                    cell3 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][2], self._filter_multiplier * 4, times=[0.5, 1, 2])
                    cell3 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier * 4, switches_cell[i][2])
                if switches_net[i][3] == [False, False, False]:
                    cell4 = None
                else:
                    prev_fmultipliers = parse_init(switches_net[i][3], self._filter_multiplier * 8, times=[0.5, 1, 2])
                    cell4 = cell(self._group_multiplier, *prev_fmultipliers, self._filter_multiplier * 8, switches_cell[i][3])
                self.cells += [cell1]
                self.cells += [cell2]
                self.cells += [cell3]
                self.cells += [cell4]
                del cell1, cell2, cell3, cell4

        self.predict = nn.Conv2d(self._filter_multiplier * self._group_multiplier, self._num_classes, 1, 1, 0, bias=True)
        self.preprocess_up_8 = ConvNorm(self._filter_multiplier * 2 * self._group_multiplier,
                      self._filter_multiplier * 1 * self._group_multiplier, 1, 1, 0, bias=True, groups=1, slimmable=False)
        self.preprocess_up_16 = ConvNorm(self._filter_multiplier * 4 * self._group_multiplier,
                      self._filter_multiplier * 2 * self._group_multiplier, 1, 1, 0, bias=True, groups=1, slimmable=False)
        self.preprocess_up_32 = ConvNorm(self._filter_multiplier * 8 * self._group_multiplier,
                      self._filter_multiplier * 4 * self._group_multiplier, 1, 1, 0, bias=True, groups=1, slimmable=False)

        self.fuse32_16 = ConvNorm(self._filter_multiplier * 8 * self._group_multiplier,
                      self._filter_multiplier * 4 * self._group_multiplier, 3, 1, 1, bias=True, groups=1, slimmable=False)
        self.fuse16_8 = ConvNorm(self._filter_multiplier * 4 * self._group_multiplier,
                      self._filter_multiplier * 2 * self._group_multiplier, 3, 1, 1, bias=True, groups=1, slimmable=False)
        self.fuse8_4 = ConvNorm(self._filter_multiplier * 2 * self._group_multiplier,
                      self._filter_multiplier * 1 * self._group_multiplier, 3, 1, 1, bias=True, groups=1, slimmable=False)

    def forward(self, x):
        #TODO: GET RID OF THESE LISTS, we dont need to keep everything.
        #TODO: Is this the reason for the memory issue ?
        self.level_4 = []
        self.level_8 = []
        self.level_16 = []
        self.level_32 = []
        temp = self.stem0(x)
        temp = self.stem1(temp)
        self.level_4.append(self.stem2(temp))
        del temp

        count = 0
        for layer in range(self._num_layers):
            if layer == 0:
                if self._switches_net[layer][0] == [False, False, False]:
                    level4_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][0], possible_inputs=[None, self.level_4[-1], None])
                    level4_new, = self.cells[count](*prev_features)
                count += 1
                if self._switches_net[layer][1] == [False, False, False]:
                    level8_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][1], possible_inputs=[self.level_4[-1], None, None])
                    level8_new, = self.cells[count](*prev_features)
                count += 1
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                del level4_new, level8_new

            elif layer == 1:
                if self._switches_net[layer][0] == [False, False, False]:
                    level4_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][0], possible_inputs=[None, self.level_4[-1], self.level_8[-1]])
                    level4_new_list = self.cells[count](*prev_features)
                    level4_new = sum(f for f in level4_new_list)
                    del level4_new_list
                count += 1
                if self._switches_net[layer][1] == [False, False, False]:
                    level8_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][1], possible_inputs=[self.level_4[-1], self.level_8[-1], None])
                    level8_new_list = self.cells[count](*prev_features)
                    level8_new = sum(f for f in level8_new_list)
                    del level8_new_list
                count += 1
                if self._switches_net[layer][2] == [False, False, False]:
                    level16_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][2], possible_inputs=[self.level_8[-1], None, None])
                    level16_new, = self.cells[count](*prev_features)
                count += 1
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)
                del level4_new, level8_new, level16_new

            elif layer == 2:
                if self._switches_net[layer][0] == [False, False, False]:
                    level4_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][0], possible_inputs=[None, self.level_4[-1], self.level_8[-1]])
                    level4_new_list = self.cells[count](*prev_features)
                    level4_new = sum(f for f in level4_new_list)
                    del level4_new_list
                count += 1
                if self._switches_net[layer][1] == [False, False, False]:
                    level8_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][1], possible_inputs=[self.level_4[-1], self.level_8[-1], self.level_16[-1]])
                    level8_new_list = self.cells[count](*prev_features)
                    level8_new = sum(f for f in level8_new_list)
                    del level8_new_list
                count += 1
                if self._switches_net[layer][2] == [False, False, False]:
                    level16_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][2], possible_inputs=[self.level_8[-1], self.level_16[-1], None])
                    level16_new_list = self.cells[count](*prev_features)
                    level16_new = sum(f for f in level16_new_list)
                    del level16_new_list
                count += 1
                if self._switches_net[layer][3] == [False, False, False]:
                    level32_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][3], possible_inputs=[self.level_16[-1], None, None])
                    level32_new, = self.cells[count](*prev_features)
                count += 1
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)
                self.level_32.append(level32_new)
                del level4_new, level8_new, level16_new, level32_new

            else:
                if self._switches_net[layer][0] == [False, False, False]:
                    level4_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][0], possible_inputs=[None, self.level_4[-1], self.level_8[-1]])
                    level4_new_list = self.cells[count](*prev_features)
                    level4_new = sum(f for f in level4_new_list)
                    del level4_new_list
                count += 1
                if self._switches_net[layer][1] == [False, False, False]:
                    level8_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][1], possible_inputs=[self.level_4[-1], self.level_8[-1], self.level_16[-1]])
                    level8_new_list = self.cells[count](*prev_features)
                    level8_new = sum(f for f in level8_new_list)
                    del level8_new_list
                count += 1
                if self._switches_net[layer][2] == [False, False, False]:
                    level16_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][2], possible_inputs=[self.level_8[-1], self.level_16[-1], self.level_32[-1]])
                    level16_new_list = self.cells[count](*prev_features)
                    level16_new = sum(f for f in level16_new_list)
                    del level16_new_list
                count += 1
                if self._switches_net[layer][3] == [False, False, False]:
                    level32_new = None
                else:
                    prev_features = parse_forward(self._switches_net[layer][3], possible_inputs=[self.level_16[-1], self.level_32[-1], None])
                    level32_new_list = self.cells[count](*prev_features)
                    level32_new = sum(f for f in level32_new_list)
                    del level32_new_list
                count += 1
                self.level_4.append(level4_new)
                self.level_8.append(level8_new)
                self.level_16.append(level16_new)
                self.level_32.append(level32_new)
                del level4_new, level8_new, level16_new, level32_new

            self.level_4 = self.level_4[-1:]
            self.level_8 = self.level_8[-1:]
            self.level_16 = self.level_16[-1:]
            self.level_32 = self.level_32[-1:]

        result_32 = F.interpolate(self.preprocess_up_32(self.level_32[-1]), size=(self.level_16[-1].shape[2], self.level_16[-1].shape[3]),
                                  mode='bilinear', align_corners=False)
        result32_16 = self.fuse32_16(torch.cat([self.level_16[-1], result_32], dim=1))
        del self.level_32[-1], self.level_16[-1], result_32
        result_16 = F.interpolate(self.preprocess_up_16(result32_16), size=(self.level_8[-1].shape[2], self.level_8[-1].shape[3]),
                                  mode='bilinear', align_corners=False)
        result16_8 = self.fuse16_8(torch.cat([self.level_8[-1], result_16], dim=1))
        del result32_16, self.level_8[-1], result_16
        result_8 = F.interpolate(self.preprocess_up_8(result16_8), size=(self.level_4[-1].shape[2], self.level_4[-1].shape[3]),
                                 mode='bilinear', align_corners=False)
        result8_4 = self.fuse8_4(torch.cat([self.level_4[-1], result_8], dim=1))
        del result16_8, self.level_4[-1], result_8
        result_4 = F.interpolate(self.predict(result8_4), size=(x.shape[2], x.shape[3]),
                                 mode='bilinear', align_corners=False)
        # print(result_4.shape)

        return result_4

    def weight_parameters(self):
        return [param for name, param in self.named_parameters()]

    # def _loss(self, input, target, aux_input):
    #     logits = self(input)
    #     return self._criterion(logits, target, aux_input)


def main():
    from decode import get_new_network
    switches_net, switches_cell = get_new_network()
    model = AutoDeeplab(19, 12, filter_multiplier=8, group_multiplier=8, criterion=None,
                        switches_cell=switches_cell, switches_net=switches_net)
    x = torch.tensor(torch.ones(4, 3, 321, 321))
    y = model(x)
    print(y.shape)
    stat(model, (3, 512, 1024))


if __name__ == '__main__' :
    main()
