import torch
import torch.nn as nn
import torchvision.models as models

from vector_quantize_pytorch import VectorQuantize

import numpy as np


def get_num_params(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class MobileNet100(nn.Module):
    def __init__(self, encdec, qbit=-1):
        super().__init__()
        self.encoder = encdec['encoder']
        self.decoder = encdec['decoder']
        self.qbit = qbit

    def quantize(self, x):
        self.max = x.max()
        self.min = x.min()
        return torch.round((2 ** self.qbit - 1) * (x - self.min) / (self.max - self.min) - 0.5)

    def dequantize(self, x):
        return x * (self.max - self.min) / (2 ** self.qbit - 1) + self.min

    def forward(self, X):
        X = self.encoder(X)
        if self.qbit != -1:
            X = self.quantize(X)
            X = self.dequantize(X)
        return self.decoder(X)


def EnsembleNet(res_stop=5, ncls=10, qbit=8):
    cfg = [(1, 16, 1, 1),
           (6, 24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6, 32, 3, 1),
           (6, 64, 4, 2),
           (6, 96, 3, 1),
           (6, 160, 3, 1),
           (6, 320, 1, 1)]
    mobilenet_v2 = models.mobilenet_v2(num_classes=1000, width_mult=1.0, inverted_residual_setting=cfg)

    mobilenet_v2.load_state_dict(torch.load('mobilenet_v2-b0353104.pth'), strict=False)

    encoder_layers = []
    decoder_layers = []

    X = torch.rand(size=(2, 3, 32, 32))

    for layer_idx, l in enumerate(mobilenet_v2.features):
        X = l(X)
        print(l.__class__.__name__, 'Output shape:\t', X.shape)
        if layer_idx <= res_stop:
            encoder_layers.append(l)
        else:
            decoder_layers.append(l)

    dropout = nn.Dropout(0.2, inplace=True)
    fc = nn.Linear(in_features=1280, out_features=ncls, bias=True)
    classifier = nn.Sequential(dropout, fc)
    pool = nn.AdaptiveAvgPool2d(1)
    decoder_layers.append(pool)
    decoder_layers.append(nn.Flatten())
    decoder_layers.append(classifier)

    print(len(encoder_layers), len(decoder_layers))

    EncDec_dict = dict(encoder=nn.Sequential(*encoder_layers), decoder=nn.Sequential(*decoder_layers))

    net = MobileNet100(EncDec_dict, qbit=qbit)
    print("Num of Parameters:", get_num_params(net))

    return net


if __name__ == "__main__":
    # vq = VectorQuantize(
    #     dim=256,
    #     codebook_size=512,  # codebook size
    #     decay=0.8,  # the exponential moving average decay, lower means the dictionary will change faster
    #     commitment_weight=1.  # the weight on the commitment loss
    # )
    #
    # x = torch.randn(1, 32, 32, 256)
    # quantized, indices, commit_loss = vq(x)  # (1, 1024, 256), (1, 1024), (1)
    #
    # print(quantized.shape, indices.shape)
    #
    # print(get_num_params(vq))

    net = EnsembleNet(res_stop=5, qbit=8).cuda()
    net.eval()
    X = torch.rand(size=(2, 3, 32, 32)).cuda()
    X = net.encoder(X)
    print(X.shape)
    X = net.quantize(X)
    print(X.shape)

    # import torch.profiler as profiler
    # with profiler.profile(
    #         activities=[
    #             profiler.ProfilerActivity.CPU,
    #             profiler.ProfilerActivity.CUDA,
    #         ],
    #         schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    #         record_shapes=True
    # ) as p:
    #     with profiler.record_function("model_inference"):
    #         a = net.encoder(X)
    # print(a.shape)
    # print(p.key_averages().table(
    #     sort_by="self_cuda_time_total", row_limit=-1))
