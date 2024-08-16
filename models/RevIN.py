import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, device="cuda"):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()
        self.to(device)

    def forward(self, x, mode: str, mask=None):
        if mode == "norm":
            self._get_statistics(x, mask)  # Set mean and std
            x = self._normalize(x, mask)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        # trainable
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x, mask):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if mask is None:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
            self.stdev = torch.sqrt(
                torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
            ).detach()
        else:
            self.mean = (
                torch.sum(x, dim=dim2reduce, keepdim=True)
                / torch.sum(mask, dim=dim2reduce, keepdim=True).detach()
            )
            x_cent = x - self.mean
            x_cent = x_cent.masked_fill(mask == 0, 0)  # reset the masked values to 0
            self.stdev = torch.sqrt(
                torch.sum(x_cent * x_cent, dim=dim2reduce, keepdim=True)
                / torch.sum(mask, dim=dim2reduce, keepdim=True)
                + self.eps
            ).detach()

    def _normalize(self, x, mask):
        x = x - self.mean
        if mask is not None:
            x = x.masked_fill(mask == 0, 0)  # reset the masked values to 0
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x