import pickle

import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader

from ensemble_model import vqee
from Utils.transforms import get_test_transform

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

num_users = 16
batch_size = 64

testset = datasets.CIFAR100(root='./data', train=False, download=True, transform=get_test_transform())
testloader = DataLoader(testset, batch_size=batch_size,
                        shuffle=False, num_workers=1, drop_last=True)

model = vqee(primary_loss=nn.CrossEntropyLoss(), n_embed=4096, skip_quant=False, width=1., commitment=0.1,
             n_ensemble=num_users).to(device)
with open('encoder.pkl', 'rb') as f:
    model.encoder = pickle.load(f)
with open('quantizer.pkl', 'rb') as f:
    model.quantizer = pickle.load(f)
with open('decoders.pkl', 'rb') as f:
    model.decoders = pickle.load(f)

entries = len(testloader)
y_hat_tensor = torch.empty([num_users, entries, batch_size, 100])
ensemble_y_hat = torch.empty([num_users, entries, batch_size, 100])


def accuracy(y, ensemble_y_pred):
    ens_pred = torch.max(ensemble_y_pred.data, 1)[1]
    return (ens_pred == y).sum()


ensemble_accuracy_per_users = torch.empty([num_users, entries])
accuracy_ensemble_tensor = torch.empty([num_users])


for num_of_ens in range(num_users):
    # if num_of_ens == 0:
    #     model.skip_quant = True
    #     model.decoder_idx = -1
    # else:
    model.decoder_idx = num_of_ens
    model.update_decoder()
    model.to(device)
    model.eval()
    print(num_of_ens)
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(testloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            _, _, y_hat_tensor[num_of_ens, b, :, :] = model.process_batch(X_test, y_test)

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

for i in range(num_users):
    print(f'Accuracy of Ensemble of {i + 1} Models: {accuracy_ensemble_tensor[i].item():.3f}%')
