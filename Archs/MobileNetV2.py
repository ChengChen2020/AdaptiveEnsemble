import copy
import numpy as np


import torch
from torch import nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights


def weight_noise(m):
    # Reset all the parameters of the new 'Decoder'.
    # For creating an ensembles of decoders.
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data = m.weight.data + torch.randn(m.weight.shape) * 0.05
        if m.bias is not None:
            m.bias.data = m.bias.data + torch.randn(m.bias.shape) * 0.02

# def get_num_params(net):
#     '''Return the number of parameters of the net'''
#     assert(isinstance(net, torch.nn.Module))
#     num = 0
#     for p in list(net.parameters()):
#         n = 1
#         for s in list(p.size()):
#             n *= s
#         num += n
#     return num

def get_num_params(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

def SplitEffNet(width=1., weights=MobileNet_V2_Weights.IMAGENET1K_V2, num_classes=10, decoder_copies=16):
    encoder_layers = []
    decoder_layers = []

    inverted_residual_setting = [[1, 16, 1, 1],
                                 [6, 24, 2, 1],
                                 [6, 32, 3, 1],
                                 [6, 64, 4, 2],
                                 [6, 96, 3, 1],
                                 [6, 160, 3, 2],
                                 [6, 320, 1, 1]]
    # num_channels_per_layer = [32, 16, 24, 24, 32, 32, 32, 64, 64, 64, 64, 96, 96, 96, 160, 160, 160, 320, 1280]

    mobilenet_v2 = models.mobilenet_v2(pretrained=True, num_classes=1000, width_mult=1.0, inverted_residual_setting=inverted_residual_setting)

    print(get_num_params(mobilenet_v2.features))

    # mobilenet_v2.load_state_dict(torch.load('mobilenet_v2-7ebf99e0.pth'), strict=False)

    X = torch.rand(size=(2, 3, 32, 32))

    res_stop = 5
    for layer_idx, l in enumerate(mobilenet_v2.features):
        # X = l(X)
        # print(l.__class__.__name__, 'Output shape:\t', X.shape)
        if layer_idx <= res_stop:
            encoder_layers.append(l)
        else:
            decoder_layers.append(l)

    dropout = nn.Dropout(0.2, inplace=True)
    fc = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    classifier = nn.Sequential(dropout, fc)
    pool = nn.AdaptiveAvgPool2d(1)
    decoder_layers.append(pool)
    decoder_layers.append(nn.Flatten())
    decoder_layers.append(classifier)

    EncDec_dict = dict(encoder=nn.Sequential(*encoder_layers), decoders=[])
    # print(EncDec_dict['encoder'])
    EncDec_dict['decoders'] = [nn.Sequential(*decoder_layers)]  # listed for a list of decoders
    # print(EncDec_dict['decoders'][0])

    a = EncDec_dict['encoder']
    print(get_num_params(a))
    b = EncDec_dict['decoders'][0]
    print(get_num_params(b))
    print(get_num_params(a) + get_num_params(b))
    for layer_idx, l in enumerate(a):
        X = l(X)
        print(l.__class__.__name__, 'Output shape:\t', X.shape)
    print()
    for layer_idx, l in enumerate(b):
        X = l(X)
        print(l.__class__.__name__, 'Output shape:\t', X.shape)

    # Creating a list of different Decoders
    decoder_copies += 1
    while decoder_copies > 1:
        new_decoder = copy.deepcopy(nn.Sequential(*decoder_layers))
        # new_decoder.apply(weight_noise)  # ?
        EncDec_dict['decoders'].append(new_decoder)
        decoder_copies -= 1

    return EncDec_dict, EncDec_dict['encoder'](torch.zeros(1, 3, 32, 32)).shape[1]


if __name__ == '__main__':
    model, _ = SplitEffNet(width=1)
    x = torch.rand(2, 3, 32, 32)
    print(type(model['encoder']))
    print(type(model['decoders'][0]))

    enc = model['encoder']
    dec = model['decoders'][0]

    # for p in enc.parameters():
    #     print(p.dtype)

    print(len(enc))
    print(len(dec))
    print(enc(x).shape)
    print(dec(enc(x)).shape)
