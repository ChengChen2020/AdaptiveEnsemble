import time
import pickle
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.models import MobileNet_V2_Weights
from vector_quantize_pytorch import VectorQuantize

from Utils.transforms import get_cifar_train_transforms, get_test_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

epochs = 200
num_users = 16
batch_size = 64

start_time = time.time()


class MobileNet10(nn.Module):
    def __init__(self, encdec, n_embed=1024, n_parts=2, skip_quant=False, decay=0.8, commitment=1.):
        super().__init__()
        self.encoder = encdec['encoder']
        self.quant_dim = self.encoder(torch.zeros(1, 3, 32, 32)).shape[1]
        self.decoder = encdec['decoder']
        self.n_embed = n_embed
        self.n_parts = n_parts
        self.skip_quant = skip_quant
        self.decay = decay
        self.commitment = commitment
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

    def forward(self, X):
        X = self.encoder(X)
        X = X.view((X.shape[0], X.shape[2], X.shape[3], X.shape[1]))
        X, _, commit_loss = self.quantize(X)
        X = X.view((X.shape[0], X.shape[3], X.shape[1], X.shape[2]))
        return self.decoder(X), commit_loss


def train_one_epoch(model, epc, part_loader, optimizer, scheduler):
    train_losses = []
    train_acc = []

    bingos = 0
    losses = 0

    model.train()
    for batch_num, (images, labels) in enumerate(part_loader):
        bs = images.shape[0]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        y_hat, commit_loss = model(images)
        # print(y_hat.shape, labels.shape)
        loss = nn.CrossEntropyLoss()(y_hat, labels) + commit_loss
        bingos += (torch.max(y_hat, 1)[1] == labels).sum()

        # Update parameters
        loss.backward()
        optimizer.step()

        if batch_num % 32 == 0:
            print(
                f'epoch: {epc:2}  batch: {batch_num:2} [{bs * batch_num:6}/{len(part_loader.dataset)}] '
                f'total loss: {loss.item():10.8f}  \
                time = [{(time.time() - start_time) / 60}] minutes')

    scheduler.step()
    # Accuracy #
    loss = losses / len(part_loader)
    train_losses.append(loss)
    acc = 100 * bingos.item() / len(part_loader.dataset)
    train_acc.append(acc)
    # train_first_acc_matrix[epc][0] = epoch_acc[epc]

    print(f'Train (Model) Accuracy at epoch {epc + 1} is {100 * bingos.item() / len(part_loader.dataset)}%')


def test(model, idx, epc, testloader):
    test_losses = []
    test_acc = []

    num_val_correct = 0
    test_losses_val = 0

    model.eval()

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(testloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_hat, commit_loss = model(X_test)
            test_loss = nn.CrossEntropyLoss()(y_hat, y_test) + commit_loss
            test_losses_val += test_loss.item()
            num_val_correct += (torch.max(y_hat, 1)[1] == y_test).sum()

        test_losses.append(test_losses_val / b)
        # writer.add_scalar("Test Loss (Model1)", test_losses_val / b, epc)
        test_acc.append(100 * num_val_correct / len(testloader.dataset))
        # writer.add_scalar("Test Accuracy (Model1)", num_val_correct / 100, epc)

    print(f'Validation (Model{idx}) Accuracy at epoch {epc + 1} is {100 * num_val_correct / len(testloader.dataset)}%')

    return 100 * num_val_correct / len(testloader.dataset)


def main():

    """
    Create 16 sub DataLoaders
    """

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=get_cifar_train_transforms())
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=get_test_transform())

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=1, drop_last=False)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=1, drop_last=True)

    mobilenet_v2 = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2, num_classes=1000, width_mult=1.)

    encoder_layers = []
    decoder_layers = []

    res_stop = 5
    for layer_idx, l in enumerate(mobilenet_v2.features):
        # X = l(X)
        # print(l.__class__.__name__, 'Output shape:\t', X.shape)
        if layer_idx <= res_stop:
            encoder_layers.append(l)
        else:
            decoder_layers.append(l)

    dropout = nn.Dropout(0.2, inplace=True)
    fc = nn.Linear(in_features=1280, out_features=10, bias=True)
    classifier = nn.Sequential(dropout, fc)
    pool = nn.AdaptiveAvgPool2d(1)
    # decoder_layers.append(pool)
    decoder_layers.append(nn.Flatten())
    decoder_layers.append(classifier)

    EncDec_dict = dict(encoder=nn.Sequential(*encoder_layers), decoder=nn.Sequential(*decoder_layers))

    model = MobileNet10(EncDec_dict).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    cnt = 0
    best_acc = 0.0
    best_model = None

    for epc in range(100):
        train_one_epoch(model, epc, trainloader, optimizer, scheduler)
        curr_acc = test(model, 0, epc, testloader)

        if curr_acc > best_acc:
            cnt = 0
            best_acc = curr_acc
            best_model = model.state_dict()
        else:
            cnt += 1

        if cnt == 5:
            break

    torch.save(best_model, 'Best_ModelNet10.pth')
    print(best_acc)


if __name__ == '__main__':
    main()
