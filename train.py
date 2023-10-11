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

from ensemble_model import vqee
from Utils.transforms import get_cifar_train_transforms, get_test_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

epochs = 200
num_users = 16
batch_size = 64

start_time = time.time()


def train_one_epoch(model, epc, part_loader, optimizer, scheduler):
    train_losses = []
    train_acc = []

    bingos = 0
    losses = 0

    # model.train()
    for batch_num, (images, labels) in enumerate(part_loader):
        bs = images.shape[0]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        result_dict, batch_acc, y_hat = model.process_batch(images, labels)
        loss = result_dict['loss']
        losses += loss.item()
        bingos += batch_acc

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
            result_dict, test_batch_acc, y_hat = model.process_batch(X_test, y_test)
            test_loss = result_dict['loss']
            test_losses_val += test_loss.item()
            num_val_correct += test_batch_acc

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

    trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=get_cifar_train_transforms())
    testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=get_test_transform())

    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=1, drop_last=False)
    testloader = DataLoader(testset, batch_size=batch_size,
                            shuffle=False, num_workers=1, drop_last=True)

    # Bagging, 34000 shared training examples
    shared_idx = 34000
    shared_indices = np.arange(0, shared_idx)
    dataloaders = []
    for i in range(num_users):
        part_indices = np.arange(shared_idx + i * 1000, shared_idx + i * 1000 + 1000)
        part_set = torch.utils.data.Subset(trainset, np.concatenate((shared_indices, part_indices)))
        dataloaders.append(DataLoader(part_set, batch_size=64,
                                      shuffle=True, num_workers=1, drop_last=False))

    assert len(dataloaders) == 16

    """
    Model Training
    """

    model = vqee(primary_loss=nn.CrossEntropyLoss(), n_embed=4096, skip_quant=False, width=1., commitment=0.1,
                 n_ensemble=num_users).to(device)
    print("How many decoders?", len(model.decoders))

    # model.skip_quant = True
    # model.decoder_idx = 16
    # model.update_decoder()
    # model.to(device)
    #
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    #
    # cnt = 0
    # best_acc = 0.0
    # for epc in range(epochs):
    #     model.train()
    #     train_one_epoch(model, epc, trainloader, optimizer, scheduler)
    #     curr_acc = test(model, 16, epc, testloader)
    #
    #     print(curr_acc, best_acc, cnt)
    #     if curr_acc > best_acc:
    #         cnt = 0
    #         best_acc = curr_acc
    #     else:
    #         cnt += 1
    #
    #     if cnt == 10:
    #         break

    """
    Train main model
    """
    cnt = 0
    best_acc = 0.0
    best_encoder = None
    best_quantizer = None
    best_decoder = defaultdict(int)
    for epc in range(epochs):

        model.train()
        train_one_epoch(model, epc, dataloaders[0], optimizer, scheduler)
        curr_acc = test(model, 0, epc, testloader)

        print(curr_acc, best_acc, cnt)
        if curr_acc > best_acc:
            best_acc = curr_acc
            best_encoder = model.encoder.state_dict()
            best_quantizer = model.quantizer.state_dict()
            best_decoder[0] = model.decoder.state_dict()
            cnt = 0
        else:
            cnt += 1

        if cnt == 5:
            break

    model.eval()

    model.encoder.load_state_dict(best_encoder)
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.quantizer.load_state_dict(best_quantizer)
    for param in model.quantizer.parameters():
        param.requires_grad = False

    print(int(model.skip_quant))

    for i in range(1, len(dataloaders)):
        model.decoder_idx = i
        model.update_decoder()
        model.to(device)

        optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

        cnt = 0
        best_acc = 0.0
        for epc in range(epochs):
            model.decoder.train()
            train_one_epoch(model, epc, dataloaders[i], optimizer, scheduler)
            curr_acc = test(model, i, epc, testloader)

            print(curr_acc, best_acc, cnt)
            if curr_acc > best_acc:
                cnt = 0
                best_acc = curr_acc
                best_decoder[i] = model.decoder.state_dict()
            else:
                cnt += 1

            if cnt == 5:
                break

        model.decoders[i].load_state_dict(best_decoder[i])

    model.skip_quant = True
    model.decoder_idx = -1
    model.update_decoder()
    model.to(device)

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    cnt = 0
    best_acc = 0.0
    for epc in range(epochs):
        model.decoder.train()
        train_one_epoch(model, epc, trainloader, optimizer, scheduler)
        curr_acc = test(model, -1, epc, testloader)

        print(curr_acc, best_acc, cnt)
        if curr_acc > best_acc:
            cnt = 0
            best_acc = curr_acc
            best_decoder[-1] = model.decoder.state_dict()
        else:
            cnt += 1

        if cnt == 5:
            break

    with open('encoder.pkl', 'wb') as f:
        pickle.dump(model.encoder, f)
    with open('decoders.pkl', 'wb') as f:
        pickle.dump(model.decoders, f)
    with open('quantizer.pkl', 'wb') as f:
        pickle.dump(model.quantizer, f)


if __name__ == '__main__':
    main()
