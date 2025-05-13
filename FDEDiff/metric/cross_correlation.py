'''


source code: https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/cross_correlation.py
'''

import torch
from torch import nn


def normalize(x, dim):
    x_mean = x.mean(dim, keepdims=True)
    x_std = x.std(dim, keepdims=True)
    x_std[x_std == 0] = 1
    return (x - x_mean) / x_std


def acf_torch(x, max_lag, dim=(0, 1)):
    D = x.shape[-1]
    x = normalize(x, dim)
    acf_list = list()
    for i in range(max_lag):
        y = x[:, i:] * x[:, :-i] if i > 0 else x * x
        acf_i = torch.mean(y, (1))
        acf_list.append(acf_i)
    acf = torch.cat(acf_list, 1)
    return acf.reshape(acf.shape[0], -1, D)


def cacf_torch(x, max_lag, dim=(0, 1)):
    def get_lower_triangular_indices(n):
        return [list(x) for x in torch.tril_indices(n, n)]

    ind = get_lower_triangular_indices(x.shape[2])
    x = normalize(x, dim)
    x_l = x[..., ind[0]]
    x_r = x[..., ind[1]]
    cacf_list = list()
    for i in range(max_lag):
        y = x_l[:, i:] * x_r[:, :-i] if i > 0 else x_l * x_r
        cacf_i = torch.mean(y, (1))
        cacf_list.append(cacf_i)
    cacf = torch.cat(cacf_list, 1)
    return cacf.reshape(cacf.shape[0], -1, len(ind[0]))


class Loss(nn.Module):
    def __init__(self, name, reg=1.0, transform=lambda x: x, threshold=10., backward=False, norm_foo=lambda x: x):
        super(Loss, self).__init__()
        self.name = name
        self.reg = reg
        self.transform = transform
        self.threshold = threshold
        self.backward = backward
        self.norm_foo = norm_foo

    def forward(self, x_fake):
        self.loss_componentwise = self.compute(x_fake)
        return self.reg * self.loss_componentwise.mean()

    def compute(self, x_fake):
        raise NotImplementedError()

    @property
    def success(self):
        return torch.all(self.loss_componentwise <= self.threshold)


class CrossCorrelLoss(Loss):
    def __init__(self, x_real, **kwargs):
        super(CrossCorrelLoss, self).__init__(norm_foo=lambda x: torch.abs(x).sum(0), **kwargs)
        self.cross_correl_real = cacf_torch(self.transform(x_real), 1).mean(0)[0]

    def compute(self, x_fake):
        cross_correl_fake = cacf_torch(self.transform(x_fake), 1).mean(0)[0]
        loss = self.norm_foo(cross_correl_fake - self.cross_correl_real)
        return loss / 10.

class AutoCorrelLoss(Loss):
    def __init__(self, x_real, max_lag, **kwargs):
        super(AutoCorrelLoss, self).__init__(norm_foo=lambda x: torch.abs(x).mean(), **kwargs)
        self.auto_correl_real = acf_torch(self.transform(x_real), max_lag).mean(0)
        self.max_lag = max_lag
    
    def compute(self, x_fake):
        auto_correl_fake = acf_torch(self.transform(x_fake), self.max_lag).mean(0)
        loss = self.norm_foo(auto_correl_fake - self.auto_correl_real)
        return loss

def Cross_CorrScore(ori_data, generated_data):
    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(generated_data)
    corr = CrossCorrelLoss(x_real, name='CrossCorrelLoss')
    return corr.compute(x_fake).item()

def Auto_CorrScore(ori_data, generated_data, max_lag):
    x_real = torch.from_numpy(ori_data)
    x_fake = torch.from_numpy(generated_data)
    corr = AutoCorrelLoss(x_real, max_lag, name='AutoCorrelLoss')
    return corr.compute(x_fake).item()