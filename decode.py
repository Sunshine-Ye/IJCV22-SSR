
def get_new_network():

    weights_nets = [[1., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [1., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0.],
                    [0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 1.],
                    [0., 0., 1., 0., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 1., 0.],
                    [0., 0., 0., 0., 1.]]

    # # global threshold
    # threshold_nets = 0.8
    # switches_net = []
    # for layer in range(len(weights_nets)):
    #     temp_layer = []
    #     for choose in range(len(weights_nets[layer])):
    #         temp_layer.append(weights_nets[layer][choose] > threshold_nets)
    #     switches_net.append(temp_layer)
    #
    # # remove "deaded" cell
    # switches_net_new = []
    # for layer in range(len(weights_nets)):
    #     temp_layer = []
    #     for choose in range(len(weights_nets[layer])):
    #         temp_layer.append(False)
    #     switches_net_new.append(temp_layer)
    #
    # for layer in range(len(switches_net)):
    #     reverted_layer = len(switches_net)-1-layer
    #     if switches_net[reverted_layer] == [False, False, False, False, False, False]:
    #         switches_net_new[reverted_layer] = [True, False, False, False, False, False]
    #     else:
    #         switches_net_new[reverted_layer] = switches_net[reverted_layer]

    # max
    switches_net_new = []
    for layer in range(len(weights_nets)):
        temp = sorted(weights_nets[layer], reverse=True)
        temp_layer = []
        for choose in range(len(weights_nets[layer])):
            # temp_layer.append(weights_nets[layer][choose] >= temp[1] and weights_nets[layer][choose] >= 0.5)
            # if choose == 0:
            #     temp_layer.append(True)
            # else:
            #     temp_layer.append(weights_nets[layer][choose] >= temp[3])
            temp_layer.append(weights_nets[layer][choose] >= temp[0])
        switches_net_new.append(temp_layer)

    print(switches_net_new)
    return switches_net_new


if __name__ == '__main__' :
    from erfnet_retrain import Encoder_retrain, Net
    from torchstat import stat
    NUM_CLASSES = 20  # pascal=22, cityscapes=20
    switches_cell = get_new_network()
    model = Encoder_retrain(NUM_CLASSES, switches_cell)
    model1 = Net(NUM_CLASSES)
    stat(model, (3, 512, 1024))
