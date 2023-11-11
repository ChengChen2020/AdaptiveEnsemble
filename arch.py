import torch
import torch.nn as nn
import torchvision.models as models

from vector_quantize_pytorch import VectorQuantize

import numpy as np


def get_num_params(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class MobileNet100(nn.Module):
    def __init__(self, encdec, n_embed=1024, n_parts=2, skip_quant=False, decay=0.8, commitment=1.0):
        super().__init__()
        self.encoder = encdec['encoder']
        self.quant_dim = self.encoder(torch.zeros(1, 3, 32, 32)).shape[1]
        self.decoder = encdec['decoder']
        self.n_embed = n_embed
        self.n_parts = n_parts
        self.skip_quant = skip_quant
        self.decay = decay
        self.commitment = commitment
        if not self.skip_quant:
            self.quantizer = VectorQuantize(dim=self.quant_dim // self.n_parts,
                                            codebook_size=self.n_embed,  # size of the dictionary
                                            decay=self.decay,  # the exponential moving average decay, lower means the
                                            # dictionary will change faster
                                            commitment_weight=self.commitment)

    def quantize(self, z_e):
        if not self.skip_quant:
            z_e_split = torch.split(z_e, self.quant_dim // self.n_parts, dim=3)
            z_q_split, indices_split = [], []
            commit_loss = 0
            for z_e_part in z_e_split:
                a, b, c, d = z_e_part.shape
                # print(z_e_part.shape)
                z_q_part, indices_part, commit_loss_part = self.quantizer(
                    z_e_part.reshape(a, -1, d)
                    # z_e_part
                )
                commit_loss += commit_loss_part
                z_q_split.append(z_q_part.reshape(a, b, c, d))
                indices_split.append(indices_part.reshape(a, b, c))
            z_q = torch.cat(z_q_split, dim=3)
            indices = torch.stack(indices_split, dim=3)
        else:
            z_q, indices, commit_loss = z_e, None, 0
        return z_q, indices, commit_loss

    def forward(self, X):
        X = self.encoder(X)
        X = X.view((X.shape[0], X.shape[2], X.shape[3], X.shape[1]))
        X, indices, commit_loss = self.quantize(X)
        # print(X.shape, indices.shape)
        X = X.view((X.shape[0], X.shape[3], X.shape[1], X.shape[2]))
        return self.decoder(X), commit_loss


def EnsembleNet(res_stop=5, ncls=10, skip_quant=True, n_embed=4096, n_parts=1, commitment=1.0):
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

    net = MobileNet100(EncDec_dict, skip_quant=skip_quant, n_embed=n_embed, n_parts=n_parts, commitment=commitment)
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

    net = EnsembleNet(res_stop=10).cuda()
    net.eval()
    X = torch.rand(size=(2, 3, 32, 32)).cuda()
    # a = net.encoder(X)
    # print(a.shape)

    import torch.profiler as profiler
    with profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            record_shapes=True
    ) as p:
        with profiler.record_function("model_inference"):
            a = net.encoder(X)
    print(a.shape)
    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))
