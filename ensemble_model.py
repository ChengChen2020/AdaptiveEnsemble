import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize

from Archs.MobileNetV2 import SplitEffNet

torch.manual_seed(1998)


class vqee(nn.Module):
    def __init__(self, primary_loss, n_embed=1024, n_parts=2, decoder_idx=0, n_ensemble=16,
                 decay=0.8, commitment=1., eps=1e-5, skip_quant=False, learning_rate=1e-3, width=1.):
        super().__init__()
        self.model, self.quant_dim = SplitEffNet(width=width, decoder_copies=n_ensemble)
        self.encoder = self.model['encoder']
        self.decoders = self.model['decoders']
        self.decoder_idx = decoder_idx
        self.n_embed = n_embed
        self.n_parts = n_parts
        self.n_ensemble = n_ensemble
        self.decay = decay
        self.skip_quant = skip_quant
        self.primary_loss = primary_loss
        self.quantizer = VectorQuantize(dim=self.quant_dim // self.n_parts,
                                        codebook_size=self.n_embed,  # size of the dictionary
                                        decay=self.decay,  # the exponential moving average decay, lower means the
                                        # dictionary will change faster
                                        commitment_weight=commitment)
        self.decoder = self.decoders[self.decoder_idx]
        self.model['quantizer'] = self.quantizer

    def encode(self, x):
        z_e = self.encoder(x)
        z_e = z_e.view((z_e.shape[0], z_e.shape[2], z_e.shape[3], z_e.shape[1]))
        return z_e

    def quantize(self, z_e):
        if not self.skip_quant:
            z_e_split = torch.split(z_e, self.quant_dim // self.n_parts, dim=3)
            z_q_split, indices_split = [], []
            commit_loss = 0
            for z_e_part in z_e_split:
                a, b, c, d = z_e_part.shape
                z_q_part, indices_part, commit_loss_part = self.quantizer(
                    z_e_part.reshape(a, -1, d)
                )
                commit_loss += commit_loss_part
                z_q_split.append(z_q_part.reshape(a, b, c, d))
                indices_split.append(indices_part)
            z_q = torch.cat(z_q_split, dim=3)
            indices = torch.stack(indices_split, dim=2)
        else:
            z_q, indices, commit_loss = z_e, None, 0
        return z_q, indices, commit_loss

    def update_decoder(self):
        self.decoder = self.decoders[self.decoder_idx]

    def decode(self, z):
        return self.decoder(z)

    def process_batch(self, x, y):
        z_e = self.encode(x)
        z_q, indices, commit_loss = self.quantize(z_e)
        # print(z_q.shape)
        z_q = z_q.view((z_q.shape[0], z_q.shape[3], z_q.shape[1], z_q.shape[2]))
        y_hat = self.decode(z_q)

        # print(z_q.shape, y_hat.shape)
        # print(torch.max(y_hat, 1)[1])
        batch_acc = self.accuracy(y, y_hat)
        prime_loss = self.primary_loss(y_hat, y)
        result_dict = {'loss': prime_loss + commit_loss, 'preds': y_hat, 'gts': y}
        return result_dict, batch_acc, y_hat

    def accuracy(self, y, y_pred):
        return (torch.max(y_pred, 1)[1] == y).sum()


if __name__ == '__main__':
    model = vqee(primary_loss=nn.CrossEntropyLoss(), n_embed=4096, skip_quant=False, width=1.)
    x = torch.randn(3, 3, 224, 224)
    y = torch.tensor([78, 77, 77])
    print(y.shape)
    a, b, c = model.process_batch(x, y)
    print(b, c)
