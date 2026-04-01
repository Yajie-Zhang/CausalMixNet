import torch, sys, os, pdb
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, device, gamma=1.0):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = torch.tensor(gamma, dtype=torch.float32).to(device)
        self.eps = 1e-6

    #         self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)

    def forward(self, input, target):
        BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none').to(self.device)
        #         BCE_loss = self.BCE_loss(input, target)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        return F_loss.mean()


