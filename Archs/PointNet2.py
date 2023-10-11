import numpy as np

from vector_quantize_pytorch import VectorQuantize

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True, commitment=0.1):
        super(get_model, self).__init__()
        in_channel = 3 if normal_channel else 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
                                             [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
                                             [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_class)

        self.quantizer = VectorQuantize(dim=1024 // 2,
                                        codebook_size=1024,  # size of the dictionary
                                        decay=0.8,  # the exponential moving average decay, lower means the
                                        # dictionary will change faster
                                        commitment_weight=commitment)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)

        z_e_split = torch.split(x, 512, dim=1)
        z_q_split, indices_split = [], []
        commit_loss = 0
        for z_e_part in z_e_split:
            z_q_part, _, commit_loss_part = self.quantizer(
                z_e_part
            )
            commit_loss += commit_loss_part
            z_q_split.append(z_q_part)
            # indices_split.append(indices_part)
        z_q = torch.cat(z_q_split, dim=1)
        # print(z_q.shape)
        # indices = torch.stack(indices_split, dim=2)

        # print(x.shape)
        x = self.drop1(F.relu(self.bn1(self.fc1(z_q))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)
        # print(x.shape)

        return x, l3_points, commit_loss


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss


if __name__ == '__main__':
    model = get_model(num_class=40, normal_channel=False)


    def get_n_params(net):
        pp = 0
        for p in list(net.parameters()):
            pp += np.prod(list(p.size()))
        return pp


    print(get_n_params(model))

    x = torch.randn(1, 3, 4096)
    model.eval()
    print(model(x)[0].shape)
