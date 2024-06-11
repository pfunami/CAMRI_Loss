import torch
import torch.nn as nn
import torch.nn.functional as F


class CAMRI_Loss(nn.Module):
    def __init__(self, important_classes, ms, s, n_classes, input_dim):
        super(CAMRI_Loss, self).__init__()
        # important_classes expected to be a list of important class indices
        assert len(important_classes) == len(ms)
        self.important_classes = important_classes
        self.m = torch.zeros(n_classes)
        for i, m in zip(important_classes, ms):
            self.m[i] = m
        self.m = nn.Parameter(self.m, requires_grad=False)
        self.s = s
        self.n_classes = n_classes
        self.weight = nn.Parameter(torch.Tensor(input_dim, n_classes))

    def forward(self, x, labels):
        # normalize features
        x_norm = F.normalize(x, p=2, dim=1)
        # normalize weights
        W = F.normalize(self.weight, p=2, dim=0)

        logits = x_norm @ W  # calculate cos
        if self.training:
            theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))  # calculate theta
            target_logits = torch.cos(theta + self.m)  # add CAMRI

            one_hot = torch.zeros_like(logits).scatter_(1, labels.float().view(-1, 1).view(-1, 1).long(), 1)
            logits = logits * (1 - one_hot) + target_logits * one_hot

        logits *= self.s  # scaling

        return logits
      
