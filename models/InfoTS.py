import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tsaug
import time
import random

from models.pretrain_models import PretrainModel
from config.configs import InfoTSConfig


def global_infoNCE(z1, z2, pooling='max', temperature=1.0):
    if pooling == 'max':
        z1 = F.max_pool1d(z1.transpose(1, 2).contiguous(), kernel_size=z1.size(1)).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2).contiguous(), kernel_size=z2.size(1)).transpose(1, 2)
    elif pooling == 'mean':
        z1 = torch.unsqueeze(torch.mean(z1, 1), 1)
        z2 = torch.unsqueeze(torch.mean(z2, 1), 1)

    # return instance_contrastive_loss(z1, z2)
    return InfoNCE(z1, z2, temperature)


def local_infoNCE(z1, z2, pooling='max', temperature=1.0, k=16):
    #   z1, z2    B X T X D
    B = z1.size(0)
    T = z1.size(1)
    D = z1.size(2)
    crop_size = int(T / k)
    crop_leng = crop_size * k

    # random start?
    start = random.randint(0, T - crop_leng)
    crop_z1 = z1[:, start:start + crop_leng, :]
    crop_z1 = crop_z1.view(B, k, crop_size, D)

    # crop_z2 = z2[:,start:start+crop_leng,:]
    # crop_z2 = crop_z2.view(B ,k,crop_size,D)

    if pooling == 'max':
        crop_z1 = crop_z1.reshape(B * k, crop_size, D)
        crop_z1_pooling = F.max_pool1d(crop_z1.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1,
                                                                                                              2).reshape(
            B, k, D)

        # crop_z2 = crop_z2.reshape(B*k,crop_size,D)
        # crop_z2_pooling = F.max_pool1d(crop_z2.transpose(1, 2).contiguous(), kernel_size=crop_size).transpose(1, 2)

    elif pooling == 'mean':
        crop_z1_pooling = torch.unsqueeze(torch.mean(z1, 1), 1)
        # crop_z2_pooling = torch.unsqueeze(torch.mean(z2,1),1)

    crop_z1_pooling_T = crop_z1_pooling.transpose(1, 2)

    # B X K * K
    similarity_matrices = torch.bmm(crop_z1_pooling, crop_z1_pooling_T)

    labels = torch.eye(k - 1, dtype=torch.float32)
    labels = torch.cat([labels, torch.zeros(1, k - 1)], 0)
    labels = torch.cat([torch.zeros(k, 1), labels], -1)

    pos_labels = labels.to(z1.device)
    pos_labels[k - 1, k - 2] = 1.0

    neg_labels = labels.T + labels + torch.eye(k)
    neg_labels[0, 2] = 1.0
    neg_labels[-1, -3] = 1.0
    neg_labels = neg_labels.to(z1.device)

    similarity_matrix = similarity_matrices[0]

    # select and combine multiple positives
    positives = similarity_matrix[pos_labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~neg_labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:, 0].mean()

    return loss


def InfoNCE(z1, z2, temperature=1.0):
    batch_size = z1.size(0)

    features = torch.cat([z1, z2], dim=0).squeeze(1)  # 2B x T x C

    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    # features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)

    logits = logits / temperature
    logits = -F.log_softmax(logits, dim=-1)
    loss = logits[:, 0].mean()

    return loss


def totensor(x, device):
    return torch.from_numpy(x).type(torch.FloatTensor).to(device)


class cutout():
    def __init__(self, perc=0.1) -> None:
        self.perc = perc

    def __call__(self, ts):
        seq_len = ts.shape[1]
        new_ts = ts.clone()
        win_len = int(self.perc * seq_len)
        start = np.random.randint(0, seq_len - win_len - 1)
        end = start + win_len
        start = max(0, start)
        end = min(end, seq_len)
        new_ts[:, start:end, :] = 0.0
        return new_ts


class jitter:
    def __init__(self, sigma=0.3, device="cuda") -> None:
        self.sigma = sigma
        self.device = device

    def __call__(self, x):
        return x + torch.normal(mean=0., std=self.sigma, size=x.shape).to(x.device)


class scaling:
    def __init__(self, sigma=0.5, device="cuda") -> None:
        self.sigma = sigma
        self.device = device

    def __call__(self, x):
        factor = torch.normal(mean=1., std=self.sigma, size=(x.shape[0], x.shape[2])).to(self.device)
        res = torch.multiply(x, torch.unsqueeze(factor, 1))
        return res


class time_warp:
    def __init__(self, n_speed_change=100, max_speed_ratio=10, device="cuda") -> None:
        self.transform = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)
        self.device = device

    def __call__(self, x_torch):
        x = x_torch.cpu().detach().numpy()
        x_tran = self.transform.augment(x)
        return totensor(x_tran.astype(np.float32), self.device)


class magnitude_warp:

    def __init__(self, n_speed_change: int = 100, max_speed_ratio=10, device="cuda") -> None:
        self.transform = tsaug.TimeWarp(n_speed_change=n_speed_change, max_speed_ratio=max_speed_ratio)
        self.device = device

    def __call__(self, x_torch):
        x = x_torch.cpu().detach().numpy()
        x_t = np.transpose(x, (0, 2, 1))
        x_tran = self.transform.augment(x_t).transpose((0, 2, 1))
        return totensor(x_tran.astype(np.float32), self.device)


class window_slice:
    def __init__(self, reduce_ratio=0.5, diff_len=True) -> None:
        self.reduce_ratio = reduce_ratio
        self.diff_len = diff_len

    def __call__(self, x):

        # begin = time.time()
        x = torch.transpose(x, 2, 1)

        target_len = np.ceil(self.reduce_ratio * x.shape[2]).astype(int)
        if target_len >= x.shape[2]:
            return x
        if self.diff_len:
            starts = np.random.randint(low=0, high=x.shape[2] - target_len, size=(x.shape[0])).astype(int)
            ends = (target_len + starts).astype(int)
            croped_x = torch.stack([x[i, :, starts[i]:ends[i]] for i in range(x.shape[0])], 0)

        else:
            start = np.random.randint(low=0, high=x.shape[2] - target_len)
            end = target_len + start
            croped_x = x[:, :, start:end]

        ret = F.interpolate(croped_x, x.shape[2], mode='linear', align_corners=False)
        ret = torch.transpose(ret, 2, 1)
        # end = time.time()
        # old_window_slice()(x)
        # end2 = time.time()
        # print(end-begin,end2-end)
        return ret


class window_warp():
    def __init__(self, window_ratio=0.3, scales=[0.5, 2.]) -> None:
        self.window_ratio = window_ratio
        self.scales = scales

    def __call__(self, x_torch):
        begin = time.time()
        B, T, D = x_torch.size()
        x = torch.transpose(x_torch, 2, 1)
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        warp_scales = np.random.choice(self.scales, B)
        warp_size = np.ceil(self.window_ratio * T).astype(int)
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(low=1, high=T - warp_size - 1, size=(B)).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        rets = []

        for i in range(x.shape[0]):
            window_seg = torch.unsqueeze(x[i, :, window_starts[i]:window_ends[i]], 0)
            window_seg_inter = \
                F.interpolate(window_seg, int(warp_size * warp_scales[i]), mode='linear', align_corners=False)[0]
            start_seg = x[i, :, :window_starts[i]]
            end_seg = x[i, :, window_ends[i]:]
            ret_i = torch.cat([start_seg, window_seg_inter, end_seg], -1)
            ret_i_inter = F.interpolate(torch.unsqueeze(ret_i, 0), T, mode='linear', align_corners=False)
            rets.append(ret_i_inter)

        ret = torch.cat(rets, 0)
        ret = torch.transpose(ret, 2, 1)
        # end = time.time()
        # old_window_warp()(x_torch)
        # end2 = time.time()
        # print(end-begin,end2-end)
        return ret


class subsequence:
    def __init__(self) -> None:
        pass

    def __call__(self, x):
        ts = x
        seq_len = ts.shape[1]
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2, high=ts_l + 1)
        new_ts = ts.clone()
        start = np.random.randint(ts_l - crop_l + 1)
        end = start + crop_l
        start = max(0, start)
        end = min(end, seq_len)
        new_ts[:, :start, :] = 0.0
        new_ts[:, end:, :] = 0.0
        return new_ts


class AutoAUG(nn.Module):
    def __init__(self, aug_p1=0.2, aug_p2=0.0, used_augs=None, device=None, dtype=None) -> None:
        super(AutoAUG, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.device = device

        all_augs = [subsequence(), cutout(), jitter(device=device), scaling(device=device),
                    time_warp(device=device), window_slice(), window_warp()]

        if used_augs is not None:
            self.augs = []
            for i in range(len(used_augs)):
                if used_augs[i]:
                    self.augs.append(all_augs[i])
        else:
            self.augs = all_augs
        self.weight = nn.Parameter(torch.empty((2, len(self.augs)), **factory_kwargs))
        self.reset_parameters()
        self.aug_p1 = aug_p1
        self.aug_p2 = aug_p2

    def get_sampling(self, temperature=1.0, bias=0.0):

        if self.training:
            bias = bias + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(self.weight.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(self.device)
            gate_inputs = (gate_inputs + self.weight) / temperature
            # para = torch.sigmoid(gate_inputs)
            para = torch.softmax(gate_inputs, -1)
            return para
        else:
            return torch.softmax(self.weight, -1)

    def reset_parameters(self) -> None:
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.01)

    def forward(self, xt):
        x, t = xt
        if self.aug_p1 == 0 and self.aug_p2 == 0:
            return x.clone(), x.clone()
        para = self.get_sampling()

        if random.random() > self.aug_p1 and self.training:
            aug1 = x.clone()
        else:
            xs1_list = []
            for aug in self.augs:
                xs1_list.append(aug(x))
            xs1 = torch.stack(xs1_list, 0)
            xs1_flattern = torch.reshape(xs1, (xs1.shape[0], xs1.shape[1] * xs1.shape[2] * xs1.shape[3]))
            aug1 = torch.reshape(torch.unsqueeze(para[0], -1) * xs1_flattern, xs1.shape)
            aug1 = torch.sum(aug1, 0)

        aug2 = x.clone()

        return aug1, aug2


class InfoTS(PretrainModel):
    def __init__(self, config: InfoTSConfig):
        super(InfoTS, self).__init__(config)
        self.n_epochs = 0
        self.n_iters = 0
        self.aug = AutoAUG(aug_p1=config.aug_p1, aug_p2=config.aug_p2, used_augs=config.used_augs, device=config.device)
        self.meta_optimizer = torch.optim.AdamW(self.aug.parameters(), lr=config.meta_lr)
        self.pred = torch.nn.Linear(self.hidden_dim, config.pretrain_batch_size)
        self.cls_optimizer = torch.optim.AdamW(self.pred.parameters(), lr=config.meta_lr)
        self.meta_lr = config.meta_lr
        self.config = config
        # contrarive between aug and original
        self.single = (config.aug_p2 == 0.0)
        self.CE = torch.nn.CrossEntropyLoss()
        self.BCE = torch.nn.BCEWithLogitsLoss()
        self.cls_lr = config.meta_lr
        self.t0 = 2.0
        self.t1 = 0.1
        self.to(config.device)

    def forward(self, x, n_epochs=-1):
        return self.get_features(x, n_epochs)

    def get_features(self, x, n_epochs=-1):
        if n_epochs == -1:
            t = 1.0
        else:
            t = float(self.t0 * np.power(self.t1 / self.t0, (self.n_epochs + 1) / self.config.pretrain_epoch))
        a1, a2 = self.aug((x, t))
        out1 = self.encoder(self.input_embedding(a1))
        out2 = self.encoder(self.input_embedding(a2))
        return out1, out2

    def compute_loss(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     criterion) -> torch.Tensor:
        out1, out2 = self(x, n_epochs=self.n_epochs)
        loss = global_infoNCE(out1, out2) + local_infoNCE(out1, out2, k=self.config.split_number) * self.config.beta
        return loss

    def epoch_start(self, epoch):
        if (self.n_epochs + 1) % 2 == 0:
            self.logger.info("Start meta-optimizing...")
            for step, (x, _) in enumerate(self.train_loader):
                x = x.to(self.device)
                y = torch.arange(x.shape[0], dtype=torch.int64).to(self.device)

                self.meta_optimizer.zero_grad()
                outv, outx = self.get_features(x)

                zv = F.max_pool1d(outv.transpose(1, 2).contiguous(), kernel_size=outv.size(1)).transpose(1, 2)
                zx = F.max_pool1d(outx.transpose(1, 2).contiguous(), kernel_size=outx.size(1)).transpose(1, 2)

                pred_yv = torch.squeeze(self.pred(zv), 1)
                pred_yx = torch.squeeze(self.pred(zx), 1)

                MI_vy_loss = self.CE(pred_yv, y)
                MI_xy_loss = self.CE(pred_yx, y)
                vx_vy_loss = 1.0 * (MI_vy_loss + MI_xy_loss)

                vx_vy_loss.backward()
                self.meta_optimizer.step()
                self.cls_optimizer.step()

    def epoch_end(self, epoch):
        self.n_epochs += 1

    def iter_end(self, iteration):
        self.n_iters += 1
