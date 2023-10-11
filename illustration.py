import torch

# dirname = './width_1/'
dirname = ''

# test_rest_acc = torch.load(dirname + 'test_rest_acc_matrix.pt')
# print(test_rest_acc)
#
# train_rest_acc = torch.load(dirname + 'train_rest_acc_matrix.pt')
accuracy_ensemble_tensor = torch.load(dirname + 'accuracy_ensemble_tensor.pt')
# y_hat = torch.load(dirname + 'y_hat_tensor.pt')

print(accuracy_ensemble_tensor[:, -1])

