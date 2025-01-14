import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):
    def __init__(self, ignore_label=255, mode=1):
        super(SegLoss, self).__init__()
        if mode == 1:
            self.obj = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        else:
            self.obj = torch.nn.NLLLoss2d(ignore_index=ignore_label)

    def __call__(self, pred, label):
        loss = self.obj(pred, label)
        return loss


class EigLoss(nn.Module):
    def __init__(self, eig=True):
        super(EigLoss, self).__init__()
        self.eig = eig
        if self.eig:
            self.L1Loss = nn.L1Loss(reduction='mean')
        self.kld = nn.KLDivLoss(reduction='mean')

    def forward(self, f1, f2):
        f1_softmax = F.softmax(f1, dim=1)
        f2_softmax = F.softmax(f2, dim=1)
        f1_log_softmax = F.log_softmax(f1, dim=1)
        loss2 = self.kld(f1_log_softmax, f2_softmax)

        if self.eig:
            loss1 = self.L1Loss(torch.diagonal(f1_softmax, dim1=-2, dim2=-1).sum(-1),
                                torch.diagonal(f2_softmax, dim1=-2, dim2=-1).sum(-1))
            loss = 1e-2 * loss1 + loss2
        else:
            loss = loss2
        return loss


class PredConLoss(nn.Module):
    def __init__(self):
        super(PredConLoss, self).__init__()
        self.obj = torch.nn.CrossEntropyLoss()

    def forward(self, f1, f2):
        f1_softmax = F.softmax(f1, dim=1)
        f2_softmax = F.softmax(f2, dim=1)

        loss = self.obj(f1_softmax, f2_softmax)
        return loss

