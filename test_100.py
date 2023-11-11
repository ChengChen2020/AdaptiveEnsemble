import torch
import torch.utils.data
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision.models as models
# import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

# import os
import time
import argparse
# import numpy as np


from arch import EnsembleNet


parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Testing')
parser.add_argument('--pp', default=5, type=int, help='partition point')
parser.add_argument('--n_parts', default=8, type=int, help='number of parts')
args = parser.parse_args()


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR100(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test(net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    print('Acc: %.3f%% (%d/%d)'
          % (100. * correct / total, correct, total))


def ensemble_test(pp=5, n_parts=2):

    net = EnsembleNet(res_stop=pp, ncls=100, skip_quant=False, n_embed=4096, n_parts=n_parts).to(device)

    X = torch.rand(size=(2, 3, 32, 32)).to(device)

    start_time = time.time()
    print(net(X)[0].shape)
    print(time.time() - start_time)

    num_users = 16
    batch_size = 100
    entries = len(testloader)
    print(entries)
    y_hat_tensor = torch.empty([num_users, entries, batch_size, 100])
    ensemble_y_hat = torch.empty([num_users, entries, batch_size, 100])

    y_pred_tensor = torch.empty([num_users, entries, batch_size])
    ensemble_y_pred = torch.empty([num_users, entries, batch_size])

    ensemble_accuracy_per_users = torch.empty([num_users, entries])
    accuracy_ensemble_tensor = torch.empty([num_users])

    def accuracy(y, ensemble_y_pred):
        ens_pred = torch.max(ensemble_y_pred.data, 1)[1]
        return (ens_pred == y).sum()

    for num_of_ens in range(num_users):

        checkpoint = torch.load(f'./checkpoint/0.0001_{pp}_100_16_{num_of_ens}_4096_{n_parts}_1.0_False_AdaptE_ckpt.pth')
        # checkpoint = torch.load('./checkpoint/ckpt_74.49.pth')
        checkpoint_2 = torch.load(f'./checkpoint/0.0001_{pp}_100_16_-1_4096_{n_parts}_1.0_False_AdaptE_ckpt.pth')
        print(checkpoint['acc'])
        net.encoder.load_state_dict(checkpoint_2['encoder'])
        net.quantizer.load_state_dict(checkpoint_2['quantizer'])
        net.decoder.load_state_dict(checkpoint['decoder'])
        # best_acc = checkpoint['acc']
        # start_epoch = checkpoint['epoch']

        net.eval()

        print(num_of_ens)

        start_time = time.time()
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(testloader):
                X_test, y_test = X_test.to(device), y_test.to(device)
                # a = net(X_test)[0].max(1)[1]
                # print(a.shape, y_test.shape)

                y_hat_tensor[num_of_ens, b, :, :], _ = net(X_test)

                # y_pred_tensor[num_of_ens, b, :] = net(X_test)[0].max(1)[1]
        print(time.time() - start_time)

    # for b, (X_test, y_test) in enumerate(testloader):
    #     for num_of_ens in range(num_users):
    #         preds = y_pred_tensor[:num_of_ens + 1, b, :]
    #         ensemble_y_pred[num_of_ens, b, :] = torch.mode(preds, dim=0)[0]
    #         ensemble_accuracy_per_users[num_of_ens, b] = ensemble_y_pred[num_of_ens, b, :].eq(y_test).sum().item()

    for b, (X_test, y_test) in enumerate(testloader):
        for num_of_ens in range(num_users):
            preds = y_hat_tensor[:num_of_ens + 1, :, :, :]
            ensemble_y_hat[num_of_ens, :, :, :] = (
                torch.mean(preds.view([num_of_ens + 1, -1]), dim=0).view([-1, batch_size, 100]))
            y_pred = ensemble_y_hat[num_of_ens, b, :, :]
            batch_ens_corr = accuracy(y_test, y_pred)
            ensemble_accuracy_per_users[num_of_ens, b] = batch_ens_corr

    for num_of_ens in range(num_users):
        total_correct = ensemble_accuracy_per_users[num_of_ens, :]
        sum_of_correct = total_correct.sum()
        acc_correct = sum_of_correct / 100.
        accuracy_ensemble_tensor[num_of_ens] = acc_correct
    #
    for i in range(num_users):
        print(f'Accuracy of Ensemble of {i + 1} Models: {accuracy_ensemble_tensor[i].item():.3f}%')


if __name__ == "__main__":
    # net = EnsembleNet(ncls=100, skip_quant=False, n_embed=4096, n_parts=1)
    #
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    # X = torch.rand(size=(2, 3, 32, 32))
    # print(net(X)[0].shape)
    # net = net.to(device)
    #
    # checkpoint = torch.load('./checkpoint/0.0001_5_200_16_-1_4096_1_1.0_True_AdaptE_ckpt.pth')
    # checkpoint_2 = torch.load('./checkpoint/0.0001_5_200_16_-1_4096_1_1.0_False_AdaptE_ckpt.pth')
    # print(checkpoint['acc'])
    # net.encoder.load_state_dict(checkpoint_2['encoder'])
    # net.quantizer.load_state_dict(checkpoint_2['quantizer'])
    # net.decoder.load_state_dict(checkpoint_2['decoder'])
    # best_acc = checkpoint['acc']
    # start_epoch = checkpoint['epoch']

    ensemble_test(pp=args.pp, n_parts=args.n_parts)
