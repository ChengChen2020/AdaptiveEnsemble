# import os
import sys
# import ssl
import time
import hydra
import pickle
import numpy as np
from omegaconf import DictConfig

import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from Archs.MobileNetV2 import SplitEffNet
from torchvision.models import MobileNet_V2_Weights
from ColabInferModel import NeuraQuantModel, NeuraQuantModel2
from Utils.transforms import get_cifar_train_transforms, get_test_transform

is_debug = sys.gettrace() is not None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

skip_quant = False
print('Skip quantization training?:', skip_quant)


@hydra.main(version_base=None, config_path="Config_Files", config_name="config")
def train(cfg: DictConfig) -> None:
    print(cfg)
    print(cfg.Ensemble.Ensemble)
    # creates headline with the current data ###
    log_dir_name = f"{cfg.Dataset.Dataset.name}_{cfg.Architecture.Architecture.name}_" \
                   f"{cfg.Quantization.Quantization}_{cfg.Ensemble.Ensemble}" \
        .replace(":", '-').replace('\'', '').replace(' ', '')
    log_dir_name = log_dir_name.replace('{', '')
    log_dir_name = log_dir_name.replace('}', '')
    log_dir_name = log_dir_name.replace(',', '_')
    print(log_dir_name)

    # checking if in debug mode for much lighter network ###
    if is_debug:
        print('in debug mode!')
        training_params = cfg.Training.training_debug
    else:
        print('in run mode!')
        training_params = cfg.Training.training

    # Prepare the Data ###
    test_transform = get_test_transform()
    if 'cifar' in cfg.Dataset.Dataset['name']:
        train_transform = get_cifar_train_transforms()
    else:
        train_transform = test_transform

    trainset = datasets.CIFAR100(root=cfg.params.data_path, train=True, download=True, transform=train_transform)
    testset = datasets.CIFAR100(root=cfg.params.data_path, train=False, download=True, transform=test_transform)

    trainloader = DataLoader(trainset, batch_size=training_params.batch_size,
                             shuffle=True, num_workers=training_params.num_workers, drop_last=True)
    testloader = DataLoader(testset, batch_size=training_params.batch_size,
                            shuffle=False, num_workers=training_params.num_workers, drop_last=True)

    EncDec_dict, _ = SplitEffNet(width=cfg.Architecture.Architecture.width,
                              # weights=MobileNet_V2_Weights.IMAGENET1K_V1,
                              num_classes=cfg.Dataset.Dataset.num_classes,
                              decoder_copies=cfg.Ensemble.Ensemble.n_ensemble)

    for x, y in testloader:
        print(x.shape, y.shape)
        break

    # path_to_pretrained_dict = './EncDec_dict_quantize.pkl'
    # if skip_quant:
    #     with open(path_to_pretrained_dict, 'rb') as f:
    #         EncDec_dict = pickle.load(f)
    #         EncDec_dict['encoder'].eval()

    learning_rate = training_params.lr
    criterion = nn.CrossEntropyLoss()
    dec = EncDec_dict['decoders'][0]

    model = NeuraQuantModel(encoder=EncDec_dict['encoder'],
                            decoder=[dec],
                            primary_loss=criterion,
                            n_embed=1024,
                            n_parts=2,
                            commitment=0.1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    start_time = time.time()
    train_losses = []
    test_losses = []
    epoch_acc = []

    num_users = 16
    num_classes = 100
    epochs = 100  # set number of epochs

    entries = len(testset) // training_params.batch_size
    train_first_acc_matrix = torch.empty([epochs, 1])
    test_first_acc_matrix = torch.empty([epochs, 1])
    y_hat_tensor = torch.empty([num_users, epochs, entries, training_params.batch_size, num_classes])

    # Number of parameters
    def get_n_params(net):
        pp = 0
        for p in list(net.parameters()):
            pp += np.prod(list(p.size()))
        return pp

    print(f'\nNumber of Parameters: {get_n_params(model)}\n')
    shared_idx = 34000
    shared_indices = np.arange(0, shared_idx)
    dataloaders = []
    for i in range(16):
        part_indices = np.arange(shared_idx + i * 1000, shared_idx + i * 1000 + 1000)
        part_set = torch.utils.data.Subset(trainset, np.concatenate((shared_indices, part_indices)))
        dataloaders.append(DataLoader(part_set, batch_size=training_params.batch_size,
                                      shuffle=True, num_workers=training_params.num_workers, drop_last=True))

    assert len(dataloaders[0]) == 546

    #####################
    # Train First Model #
    #####################

    if training_params.train_first:
        part_loader = dataloaders[0]
        for epc in range(epochs):
            bingos = 0
            losses = 0
            for batch_num, (images, labels) in enumerate(part_loader):
                images, labels = images.to(device), labels.to(device)
                # print(images.shape, labels.shape)
                result_dict, batch_acc, y_hat = model.process_batch(images, labels)
                loss = result_dict['loss']
                losses += loss.item()
                bingos += batch_acc[0]

                # Update parameters
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if batch_num % 32 == 0:
                    print(
                        f'epoch: {epc:2}  batch: {batch_num:2} [{training_params.batch_size * batch_num:6}/{len(part_loader.dataset)}]  total loss: {loss.item():10.8f}  \
                    time = [{(time.time() - start_time) / 60}] minutes')

            scheduler.step()
            # Accuracy #
            loss = losses / len(part_loader)
            # writer.add_scalar("Train Loss (Model1)", loss, epc)
            train_losses.append(loss)
            accuracy = 100 * bingos.item() / len(part_loader.dataset)
            epoch_acc.append(accuracy)
            # writer.add_scalar("Train Accuracy (Model1)", accuracy, epc)
            train_first_acc_matrix[epc][0] = epoch_acc[epc]

            num_val_correct = 0
            model.eval()
            test_losses_val = 0

            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(testloader):
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    result_dict, test_batch_acc, y_hat = model.process_batch(X_test, y_test)
                    test_loss = result_dict['loss']
                    test_losses_val += test_loss.item()
                    num_val_correct += test_batch_acc[0]
                    y_hat_tensor[0, epc, b - 1, :, :] = y_hat[0]

                test_losses.append(test_losses_val / b)
                # writer.add_scalar("Test Loss (Model1)", test_losses_val / b, epc)
                test_first_acc_matrix[epc][0] = 100 * num_val_correct / len(testset)
                # writer.add_scalar("Test Accuracy (Model1)", num_val_correct / 100, epc)

            print(f'Train (Model1) Accuracy at epoch {epc + 1} is {100 * bingos.item() / len(part_loader.dataset)}%')
            print(f'Validation (Model1) Accuracy at epoch {epc + 1} is {100 * num_val_correct / len(testset)}%')
            model.train()

        # torch.save(model.encoder.state_dict(), 'NeuraQuantizerEncoder.pt')
        # torch.save(model.quantizer.state_dict(), 'NeuraQuantizerQuantizer.pt')

    EncDec_dict['encoder'].eval()
    EncDec_dict['quantizer'] = model.quantizer
    EncDec_dict['quantizer'].eval()

    for param in EncDec_dict['encoder'].parameters():
        param.requires_grad = False
    for param in EncDec_dict['quantizer'].parameters():
        param.requires_grad = False

    #####################
    # Train Next Models #
    #####################

    train_rest_acc_matrix = torch.empty([epochs, num_users - 1])
    test_rest_acc_matrix = torch.empty([epochs, num_users - 1])
    # y_hat_tensor_rest = torch.empty([num_users-1, epochs, entries, training_params.batch_size, num_classes])

    print(len(EncDec_dict['decoders']))

    for num in range(1, len(EncDec_dict['decoders'])):
        # strings for tensorboard
        model_name = 'Model' + str(num + 1)

        dec = EncDec_dict['decoders'][num]
        # Fix encoder and quantizer
        model2 = NeuraQuantModel2(encoder=EncDec_dict['encoder'],
                                  decoder=[dec],
                                  quantizer=EncDec_dict['decoder'],
                                  primary_loss=nn.CrossEntropyLoss(),
                                  n_embed=cfg.Quantization.Quantization.n_embed,
                                  n_parts=cfg.Quantization.Quantization.n_parts,
                                  commitment=cfg.Quantization.Quantization.commitment_w).to(device)

        model2.encoder.eval()
        for param in model2.encoder.parameters():
            param.requires_grad = False

        optimizer2 = torch.optim.Adam(model2.parameters(), lr=learning_rate)
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=3, gamma=0.9)

        for epc in range(epochs):
            bingos = 0
            losses = 0

            part_loader = dataloaders[num]

            for batch_num, (images, labels) in enumerate(part_loader):
                images, labels = images.to(device), labels.to(device)
                print(labels.shape)
                result_dict, batch_acc, y_hat = model2.process_batch_fixed(images, labels)
                loss = result_dict['loss']
                losses += loss.item()
                bingos += batch_acc[0]

                # Update parameters
                loss.backward()
                optimizer2.step()
                optimizer2.zero_grad()

                if batch_num % 32 == 0:
                    print(
                        f'epoch: {epc:2}  batch: {batch_num:2} [{training_params.batch_size * batch_num:6}/{len(part_loader.dataset)}]  total loss: {loss.item():10.8f}  \
                    time = [{(time.time() - start_time) / 60}] minutes')

            scheduler2.step()
            # Accuracy #
            tot_loss = losses / len(part_loader)
            train_losses.append(tot_loss)
            # writer.add_scalar(train_loss_str, tot_loss, epc)
            accuracy = 100 * bingos.item() / len(part_loader.dataset)
            # writer.add_scalar(train_acc_str, accuracy, epc)
            epoch_acc.append(accuracy)
            train_rest_acc_matrix[epc][num - 1] = epoch_acc[epc]

            num_val_correct = 0
            model2.decoder.eval()
            test_losses_val = 0

            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(testloader):
                    X_test, y_test = X_test.to(device), y_test.to(device)
                    result_dict, test_batch_acc, y_hat = model2.process_batch_fixed(X_test, y_test)
                    test_loss = result_dict['loss']
                    test_losses_val += test_loss.item()
                    num_val_correct += test_batch_acc[0]
                    y_hat_tensor[num, epc, b - 1, :, :] = y_hat[0]

                test_losses.append(test_losses_val / b)
                # writer.add_scalar(test_loss_str, test_losses_val / b, epc)
                test_rest_acc_matrix[epc][num - 1] = 100 * num_val_correct / len(testset)
                # writer.add_scalar(test_acc_str, num_val_correct / 100, epc)

            print(f'Train {model_name} Accuracy at epoch {epc + 1} is {100 * bingos / len(part_loader.dataset)}%')
            print(f'Validation {model_name} Accuracy at epoch {epc + 1} is {100 * num_val_correct / len(testset)}%')
            model2.decoder.train()

        # Calculate the Ensemble Validation Accuracy #

    torch.save(y_hat_tensor, 'y_hat_tensor.pt')

    # Save pretrained model with no quantization
    if cfg.Training.Training.un_quantized_training:
        with open('EncDec_dict.pkl', 'wb') as f:
            pickle.dump(EncDec_dict, f)

    if not cfg.Training.Training.un_quantized_training:
        with open('EncDec_dict_quantize.pkl', 'wb') as f:
            pickle.dump(EncDec_dict, f)

    def ensemble_calculator(preds_list, num_users):
        stack = preds_list.view([num_users, -1])
        return torch.mean(stack, axis=0)

    def accuracy(y, ensemble_y_pred):
        ens_pred = torch.max(ensemble_y_pred.data, 1)[1]
        batch_ens_corr = (ens_pred == y).sum()
        return batch_ens_corr

    # Calculate the predictions of the Ensembles

    ensemble_y_hat = torch.empty([num_users, epochs, entries, training_params.batch_size, num_classes])

    for num_of_ens in range(num_users):
        for epc in range(epochs):
            preds = y_hat_tensor[:num_of_ens + 1, epc, :, :, :]
            mean = ensemble_calculator(preds, num_of_ens + 1)
            mean = mean.view([-1, training_params.batch_size, num_classes])
            ensemble_y_hat[num_of_ens, epc, :, :, :] = mean

    ensemble_accuracy_per_users = torch.empty([num_users, epochs, entries])
    accuracy_ensemble_tensor = torch.empty([num_users, epochs])

    # Checking the accuracy of the ensemble predictions

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(testloader):
            if len(X_test) != training_params.batch_size:
                continue
            for num_of_ens in range(num_users):
                for epc in range(epochs):
                    y_pred = ensemble_y_hat[num_of_ens, epc, b, :, :]
                    batch_ens_corr = accuracy(y_test, y_pred)
                    ensemble_accuracy_per_users[num_of_ens, epc, b] = batch_ens_corr

        for num_of_ens in range(num_users):
            for epc in range(epochs):
                total_correct = ensemble_accuracy_per_users[num_of_ens, epc, :]
                sum_of_correct = total_correct.sum()
                acc_correct = sum_of_correct * (100 / 10000)
                accuracy_ensemble_tensor[num_of_ens, epc] = acc_correct

    # save losses to file
    torch.save(model, 'model.pt')
    torch.save(accuracy_ensemble_tensor, 'accuracy_ensemble_tensor.pt')
    torch.save(test_rest_acc_matrix, 'test_rest_acc_matrix.pt')
    torch.save(train_rest_acc_matrix, 'train_rest_acc_matrix.pt')

    for i in range(num_users):
        print(f'Accuracy of Ensemble of {i + 1} Models: {accuracy_ensemble_tensor[i, -1].item():.3f}%')

    print('\nTRAINING HAS FINISHED SUCCESSFULLY')
    print(log_dir_name)


# Train the Model #
if __name__ == '__main__':
    train()
