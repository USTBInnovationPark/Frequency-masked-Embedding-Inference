import copy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import util.utils as tool
from enum import Enum
from train.trainable import TrainableModule
from config.configs import *
from models.RevIN import RevIN

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
import sklearn.metrics as m


def get_hidden_dim(config: PretrainConfig):
    if config.encoder_size == "tiny":
        return 128
    elif config.encoder_size == "small":
        return 256
    elif config.encoder_size == "norm":
        return 512
    elif config.encoder_size == "big":
        return 1024
    else:
        raise NotImplementedError("Encoder size: {} is not implemented.".format(config.encoder_size))


def get_feature_fusion(config: PretrainConfig):
    if config.feature_fusion == "mean":
        return lambda x: x.mean(dim=-2)
    elif config.feature_fusion == "first":
        return lambda x: x[:, 0, :]
    elif config.feature_fusion == "last":
        return lambda x: x[:, -1, :]
    elif config.feature_fusion == "all":
        return lambda x: x
    elif config.feature_fusion == "max":
        return lambda x: torch.max(x, dim=-2).values
    else:
        raise NotImplementedError("Fusion method: {} is not implemented.".format(config.feature_fusion))


class PositionEmbedding(nn.Module):
    def __init__(self, dim, max_size=2000, dropout=0.5, device="cuda:0"):
        super(PositionEmbedding, self).__init__()
        self.pe = torch.zeros(max_size, dim)
        position = torch.arange(0, max_size).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) *
                             -(np.log(10000.0) / dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape = (b, l, dim)
        seq_len = x.shape[-2]
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, patch_size, patch_step):
        super(Patching, self).__init__()
        assert patch_step > 0 and patch_size > 0
        self.patch_size = patch_size
        self.patch_step = patch_step

    def forward(self, x):
        """
        :param x: shape = (Batch, Channel, length)
        :return out: shape = (Batch, patch_num, patch_length)
        """
        B = x.shape[0]
        x = x.unfold(dimension=-1,
                     size=self.patch_size,
                     step=self.patch_step).contiguous()  # x.shape = (batch, channel, patch_num, patch_size)
        x = x.view(B, x.shape[-2], -1)  # x.shape = (batch, patch_num, patch_size*channel)
        return x


class DefaultInputEmbedding(nn.Module):
    def __init__(self, config: PretrainConfig):
        super(DefaultInputEmbedding, self).__init__()

        class Transpose(nn.Module):
            def __init__(self):
                super(Transpose, self).__init__()

            def forward(self, x):
                return x.transpose(-1, -2)
        self.embedding = nn.Sequential(
            Transpose(),
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.AvgPool1d(2, 2),
            Transpose()
        )
        self.to(config.device)

    def forward(self, x):
        """
        :param x: shape = (Batch, length, 1)
        :return out: shape = (Batch, length, Dim) or (Batch, patch_num, Dim)
        """
        return self.embedding(x)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, out_act=True):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm1d(out_channel),
            nn.GELU(),
            nn.Conv1d(in_channels=out_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False),
            nn.BatchNorm1d(out_channel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels=in_channel,
                          out_channels=out_channel,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm1d(out_channel),
            )
        self.out_act = nn.GELU() if out_act else None

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = self.out_act(out) if self.out_act is not None else out
        return out


class ResNet(nn.Module):
    def __init__(self,
                 size="tiny",
                 stride=2):
        super(ResNet, self).__init__()
        self.in_channel = 16
        if size == "tiny":
            self.layer1 = self.make_layer(ResBlock, 16, 2, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 32, 2, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 64, 2, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 128, 2, stride=stride, out_act=True)
        elif size == "small":
            self.layer1 = self.make_layer(ResBlock, 32, 2, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 64, 2, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 128, 2, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 256, 2, stride=stride, out_act=True)
        elif size == "norm":
            self.layer1 = self.make_layer(ResBlock, 64, 2, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 128, 2, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 256, 2, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 512, 2, stride=stride, out_act=True)
        else:
            self.layer1 = self.make_layer(ResBlock, 128, 2, stride=stride)
            self.layer2 = self.make_layer(ResBlock, 256, 2, stride=stride)
            self.layer3 = self.make_layer(ResBlock, 512, 2, stride=stride)
            self.layer4 = self.make_layer(ResBlock, 1024, 2, stride=stride, out_act=True)

    def make_layer(self, block, channels, num_blocks, stride, out_act=True):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride=stride, out_act=out_act))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: A input series with shape (Batch, Length, 16).
        """
        x = x.transpose(-1, -2)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.transpose(-1, -2)
        return out


class PretrainModel(TrainableModule):
    def __init__(self, config: PretrainConfig, input_embedding=None):
        super(PretrainModel, self).__init__(config)
        self.input_embedding = DefaultInputEmbedding(config) if input_embedding is None else input_embedding
        self.encoder = ResNet(config.encoder_size, stride=config.stride)
        self.feature_fusion = get_feature_fusion(config)
        self.hidden_dim = get_hidden_dim(config)

    def compute_loss(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     criterion) -> torch.Tensor:
        raise NotImplementedError("The loss function behavior of the pre-train method must be implemented"
                                  "in compute_loss().")

    def train_end(self):
        ck = {"input_embedding": self.input_embedding.state_dict(),
              "encoder": self.encoder.state_dict()}
        torch.save(ck, os.path.join(self.model_path, "model_ck.pt"))


class DownStreamModel(TrainableModule):
    def __init__(self,
                 config: FineTuneConfig,
                 pretrainModel: PretrainModel):
        super(DownStreamModel, self).__init__(config)
        self.input_embedding = pretrainModel.input_embedding
        self.encoder = pretrainModel.encoder
        self.feature_fusion = pretrainModel.feature_fusion
        self.hidden_dim = pretrainModel.hidden_dim
        self.encoder.requires_grad_(config.finetune_encoder)
        self.finetune_encoder = config.finetune_encoder
        if not config.finetune_encoder:
            self.input_embedding.eval()
            self.encoder.eval()
        self.input_embedding.requires_grad_(config.finetune_encoder)
        if config.finetune_encoder != config.affine_bn:
            for m in self.encoder.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.requires_grad_(config.affine_bn)
            for m in self.input_embedding.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.requires_grad_(config.affine_bn)

    def load_pretrain_model(self, pretrain_model: str or PretrainModel):
        if isinstance(pretrain_model, str):
            ck = torch.load(os.path.join(pretrain_model, "model_ck.pt"))
            self.input_embedding.load_state_dict(ck["input_embedding"])
            self.encoder.load_state_dict(ck["encoder"])
        elif isinstance(pretrain_model, PretrainModel):
            self.input_embedding.load_state_dict(pretrain_model.input_embedding.state_dict())
            self.encoder.load_state_dict(pretrain_model.encoder.state_dict())
        else:
            raise ValueError("Only model path and PretrainModel can be loaded.")

    def epoch_test_model(self):
        raise NotImplementedError()

    # def iter_end(self, iteration):
    #     print(torch.cuda.memory_summary(self.device))
    #     return

    def train_end(self):
        self.logger.info("______________________Test Result______________________")
        self.epoch_test_model()

    def epoch_start(self, epoch):
        if not self.finetune_encoder:
            self.input_embedding.eval()
            self.encoder.eval()
        else:
            self.input_embedding.train()
            self.encoder.train()



class ClassifierModel(DownStreamModel):
    def __init__(self,
                 config: DownstreamConfig_cls,
                 pretrainModel: PretrainModel):
        super(ClassifierModel, self).__init__(config, pretrainModel)
        self.classifier = nn.Linear(in_features=self.hidden_dim, out_features=config.cls_num)
        self.test_results = None
        self.to(config.device)

    def forward(self, x):
        return self.classifier(self.feature_fusion(self.encoder(self.input_embedding(x))))

    def compute_loss(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     criterion) -> torch.Tensor:
        pred = self(x)
        return self.criterion(pred, y.to(torch.long))

    def epoch_test_model(self):
        self.eval()
        labels_numpy_all, pred_numpy_all = None, None
        with torch.no_grad():
            for step, (x, y) in enumerate(self.test_loader):
                x = x.to(torch.float32).to(self.device)
                y = y.to(torch.float32).to(self.device)
                model_out = self(x)
                model_out = model_out.detach().cpu()
                pred = model_out.argmax(axis=-1)
                y = y.detach().cpu()
                if labels_numpy_all is not None:
                    labels_numpy_all = np.concatenate((labels_numpy_all, y))
                    pred_numpy_all = np.concatenate((pred_numpy_all, pred))
                else:
                    labels_numpy_all = y
                    pred_numpy_all = pred
        recall = m.recall_score(pred_numpy_all, labels_numpy_all, average="macro")
        precision = m.precision_score(pred_numpy_all, labels_numpy_all, average="macro")
        f1 = m.f1_score(pred_numpy_all, labels_numpy_all, average="macro")
        acc = m.accuracy_score(pred_numpy_all, labels_numpy_all)
        self.test_results = {"acc": acc, "precision": precision, "recall": recall, "f1": f1}
        self.logger.info("Recall: {:.4f}\tPrecision: {:.4f}\tF1: {:.4f}\tAccuracy: {:.4f}".format(recall,
                                                                                                  precision,
                                                                                                  f1,
                                                                                                  acc))


class RegressionModel(DownStreamModel):
    def __init__(self,
                 config: DownstreamConfig_pred,
                 pretrainModel: PretrainModel):
        super(RegressionModel, self).__init__(config, pretrainModel)
        self.predictor = nn.Linear(in_features=self.hidden_dim, out_features=1)
        self.to(config.device)

    def forward(self, x):
        return self.predictor(self.feature_fusion(self.encoder(self.input_embedding(x))))

    def compute_loss(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     criterion) -> torch.Tensor:
        return self.criterion(self(x), y)

    def epoch_test_model(self):
        self.eval()
        labels_numpy_all = []
        pred_numpy_all = []
        with torch.no_grad():
            for step, (x, y) in enumerate(self.test_loader):
                x = x.to(torch.float32).to(self.device)
                y = y.to(torch.float32).to(self.device)
                model_out = self(x)
                model_out = model_out.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                labels_numpy_all.append(y)
                pred_numpy_all.append(model_out)
        labels_numpy_all = np.concatenate(labels_numpy_all, axis=0)
        pred_numpy_all = np.concatenate(pred_numpy_all, axis=0)
        mse = m.mean_squared_error(labels_numpy_all, pred_numpy_all)
        mae = m.mean_absolute_error(labels_numpy_all, pred_numpy_all)
        mape = m.mean_absolute_percentage_error(labels_numpy_all, pred_numpy_all)
        self.logger.info("MSE: {:.4f}\tMAE: {:.4f}\tMAPE: {:.4f}".format(mse, mae, mape))


class FreqMaskJECNNModel(PretrainModel):
    def __init__(self,
                 config: FreqMaskJEConfig):
        super(FreqMaskJECNNModel, self).__init__(config)
        self.global_encoder = copy.deepcopy(self.encoder)
        self.global_encoder.requires_grad_(False)
        self.global_input_embedding = copy.deepcopy(self.input_embedding)
        self.global_input_embedding.requires_grad_(False)
        self.shrink_dim = self.hidden_dim // 2
        self.predictor_dim = self.shrink_dim // 2
        if config.mask_embedding:
            mask_len = config.pretrain_sample_length // 2 + 1 \
                if config.mask_type != "temporal" else config.pretrain_sample_length
            self.reduce_prompt_encoder = nn.Parameter(torch.randn(mask_len, self.shrink_dim))
        else:
            self.reduce_prompt_encoder = nn.Sequential(
                nn.Linear(in_features=config.pretrain_sample_length // 2 + 1,
                          out_features=self.shrink_dim, bias=False),
            )
        self.subspace_1 = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim, out_features=self.shrink_dim, bias=True),
        )
        self.subspace_1_copy = copy.deepcopy(self.subspace_1)
        self.subspace_1_copy.requires_grad_(False)
        self.reduce_feature_predictor = nn.Sequential(
            nn.Linear(in_features=self.shrink_dim, out_features=self.predictor_dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=self.predictor_dim, out_features=self.shrink_dim, bias=True),
        )
        self.mask_predictor = nn.Sequential(
            nn.Linear(in_features=self.shrink_dim, out_features=self.predictor_dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=self.predictor_dim, out_features=self.shrink_dim, bias=True),
        )

        self.momentum_iter = 0
        self.reduce_mask_pool = None
        self.mask_pool_size = config.pretrain_batch_size * config.mask_pool_size
        self.mask_applicator = tool.apply_freq_reduce_mask \
            if config.mask_type != "temporal" else tool.apply_temporal_mask
        self.config = config

        self.to(config.device)

        # ⬇ These variables are used to explore the behavior of the FMEI
        self.visual_samples = None
        self.masked_visual_samples = None
        self.visual_embeddings = None
        self.visual_embeddings_sub = None
        self.visual_masked_embeddings = None
        self.pred_visual_masked_embeddings = None
        self.visual_masked_embeddings_sub = None
        self.mask_test = None
        self.mask_embeddings = None
        self.writer = None
        self.visual_recs = None

        self.i_grad = []
        self.m_grad = []
        self.e_grad = []

        self.emb_pred_loss = []
        self.mask_pred_loss = []

    def train_start(self):
        if self.config.mask_aug_during_training:
            self.aug_epoch = self.config.pretrain_epoch // (
                    (self.config.reduce_mask_ratio[1] - self.config.reduce_mask_ratio[0]) // 0.05)
            self.aug_epoch //= 2
            self.logger.info("Mask Pool will be regenerated every {} epoch.".format(self.aug_epoch))
            self.generate_masks(size=self.mask_pool_size,
                                sample_len=self.config.pretrain_sample_length,
                                ratio=[self.config.reduce_mask_ratio[0], self.config.reduce_mask_ratio[0] + 0.01],
                                target_num=self.config.target_num)
        else:
            self.generate_masks(size=self.mask_pool_size,
                                sample_len=self.config.pretrain_sample_length,
                                ratio=self.config.reduce_mask_ratio,
                                target_num=self.config.target_num)
        self.logger.info("Done.")
        self.init_visual_sample()
        self.writer = SummaryWriter(log_dir=os.path.join(self.model_path, "runs"))
        self.logger.info("Visual samples are Initialized.")

    def mask_encode(self, mask):
        if self.config.mask_embedding:
            masks = 1 - mask
            reduce_mask_embed = (masks @ self.reduce_prompt_encoder) / (
                    torch.sum(masks, dim=-1, keepdim=True) ** 0.5 + 1e-6)
            # reduce_mask_embed = (masks @ self.reduce_prompt_encoder) / (
            #         (self.config.pretrain_sample_length//2+1) ** 0.5 + 1e-6)
        else:
            if self.config.dense_mask:
                masks = 2 * mask - 1
                reduce_mask_embed = self.reduce_prompt_encoder(masks)  # (B, num, D_s), all index are transposed.
            else:
                masks = 1 - mask
                reduce_mask_embed = \
                    self.reduce_prompt_encoder(masks)
        return reduce_mask_embed

    def forward(self, x):
        # sample augmentation
        B, L, C = x.shape[0], x.shape[1], x.shape[2]

        # target sample
        reduce_masks = self.select_masks(B, self.reduce_mask_pool)  # (B, num, F_L)
        reduce_masks = reduce_masks.to(self.config.device)
        reduce_targets = self.mask_applicator(x, reduce_masks)  # shape = (B, num, L, C)

        # feature extraction
        context_embed = self.feature_fusion(self.encoder(self.input_embedding(x)))  # (B, D)
        sub_embed_1 = self.subspace_1(context_embed)

        reduce_mask_embed = self.mask_encode(reduce_masks)

        reduce_targets_embed = []
        reduce_pred_embed = []
        mask_pred = []
        for i in range(self.config.target_num):
            # feature prediction process:
            reduce_target_embed = self.subspace_1_copy(
                self.feature_fusion(self.global_encoder(self.global_input_embedding(reduce_targets[:, i]))))
            reduce_pred_ = self.reduce_feature_predictor((sub_embed_1 + reduce_mask_embed[:, i, :].detach()))  # (B, D)
            reduce_pred_ = reduce_pred_.view(B, -1)
            reduce_targets_embed.append(reduce_target_embed)
            reduce_pred_embed.append(reduce_pred_)
            # mask prediction process:
            mask_pred_ = self.mask_predictor(reduce_target_embed - sub_embed_1.detach())
            mask_pred.append(mask_pred_)

        return torch.stack(reduce_targets_embed, dim=0).detach(), torch.stack(reduce_pred_embed, dim=0), \
            reduce_mask_embed, torch.stack(mask_pred, dim=1)

    def iter_end(self, iteration):
        # momentum update global encoder
        ipe = len(self.train_loader)  # iteration per epoch
        ema = self.config.exponential_moving_average_rate
        ema = ema + self.momentum_iter * (1 - ema) / (ipe * self.config.pretrain_epoch)
        self.momentum_iter += 1 if self.momentum_iter < ipe * self.config.pretrain_epoch + 1 else -self.momentum_iter
        for (param_c, param_g) in zip(self.input_embedding.parameters(), self.global_input_embedding.parameters()):
            param_g.data.mul_(ema).add_((1 - ema) * param_c.detach().data)
        for (param_c, param_g) in zip(self.encoder.parameters(), self.global_encoder.parameters()):
            param_g.data.mul_(ema).add_((1 - ema) * param_c.detach().data)
        for (param_c, param_g) in zip(self.subspace_1.parameters(), self.subspace_1_copy.parameters()):
            param_g.data.mul_(ema).add_((1 - ema) * param_c.detach().data)

    def iter_end_before_opt(self, iteration):
        # 记录梯度
        i_grad = 0  # input_embedding grad
        e_grad = 0  # encoder grad
        m_grad = 0  # mask embedding grad
        for param in self.input_embedding.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                i_grad += param_norm.item() ** 2
        for param in self.encoder.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                e_grad += param_norm.item() ** 2
        if isinstance(self.reduce_prompt_encoder, nn.Parameter):
            if self.reduce_prompt_encoder.grad is not None:
                param_norm = self.reduce_prompt_encoder.grad.data.norm(2)
                m_grad += param_norm.item() ** 2
        else:
            for param in self.reduce_prompt_encoder.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    m_grad += param_norm.item() ** 2
        self.i_grad.append(i_grad)
        self.e_grad.append(e_grad)
        self.m_grad.append(m_grad)

    def select_masks(self, size, mask_pool):
        indices = torch.randperm(mask_pool.size(0))
        return torch.index_select(mask_pool, dim=0, index=indices)[:size]

    def compute_loss(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     criterion) -> torch.Tensor:
        feature_label, feature_out, mask_label, mask_pred = self(x)
        emb_loss = self.criterion(feature_label, feature_out)
        mask_loss = self.criterion(mask_label, mask_pred)
        self.emb_pred_loss.append(emb_loss.item())
        self.mask_pred_loss.append(mask_loss.item())
        # feature_label, feature_out = self(x)
        return emb_loss + mask_loss

    def epoch_end(self, epoch):
        if self.config.mask_aug_during_training:
            if (epoch + 1) % self.aug_epoch == 0:
                min_ratio = self.config.reduce_mask_ratio[0]
                max_ratio_ada = min_ratio + (epoch + 1) // self.aug_epoch * 0.05
                if max_ratio_ada >= self.config.reduce_mask_ratio[1]:
                    max_ratio_ada = self.config.reduce_mask_ratio[1]
                self.logger.info("Mask Pool is regenerated utilizing ratio [{}, {}]".format(min_ratio, max_ratio_ada))
                self.generate_masks(self.mask_pool_size,
                                    self.config.pretrain_sample_length,
                                    [min_ratio, max_ratio_ada],
                                    self.config.target_num)
        if self.config.visual_samples:
            with torch.no_grad():
                self.visual_sample_process()
                self.visual_sample_plot(3, 7, epoch)
        # 记录梯度
        self.writer.add_scalar("Input Embedding grad Sum", np.sum(self.i_grad), epoch)
        self.writer.add_scalar("Input Embedding grad Mean", np.mean(self.i_grad), epoch)
        self.writer.add_scalar("Encoder grad Sum", np.sum(self.e_grad), epoch)
        self.writer.add_scalar("Encoder grad Mean", np.mean(self.e_grad), epoch)
        self.writer.add_scalar("Mask Encoder grad Sum", np.sum(self.m_grad), epoch)
        self.writer.add_scalar("Mask Encoder grad Mean", np.mean(self.m_grad), epoch)
        self.i_grad = []
        self.e_grad = []
        self.m_grad = []
        self.logger.info("\t Embedding Loss: {:.4f}, Mask Loss: {:.4f}".format(
            np.mean(self.emb_pred_loss),
            np.mean(self.mask_pred_loss))
        )

    def generate_masks(self, size, sample_len, ratio, target_num):
        self.logger.info("Constructing {} Mask Pool with size: {}".format(self.config.mask_type,
                                                                          size))
        if self.config.mask_type == "continue":
            self.reduce_mask_pool = tool.get_batch_continuous_freq_mask(size,
                                                                        sample_len,
                                                                        ratio,
                                                                        target_num)
        elif self.config.mask_type == "discrete":
            self.reduce_mask_pool = tool.get_batch_discrete_freq_mask(size,
                                                                      sample_len,
                                                                      ratio,
                                                                      target_num)
        elif self.config.mask_type == "temporal":
            self.reduce_mask_pool = tool.get_batch_temporal_mask(size,
                                                                 sample_len,
                                                                 ratio,
                                                                 target_num)
        else:
            raise ValueError("Unknown mask type {}.".format(self.config.mask_type))

    def init_visual_sample(self):
        self.visual_samples = self.train_loader.dataset[:1000:100][0].float()
        mask_len = self.config.pretrain_sample_length // 2 + 1 \
            if self.config.mask_type != "temporal" else self.config.pretrain_sample_length
        self.mask_test = torch.ones((self.visual_samples.shape[0], 10, mask_len))
        for i in range(10):
            self.mask_test[:, i, :2 * (i + 1)] = 0
        self.mask_test = self.mask_test.to(self.config.device)
        self.visual_samples = self.visual_samples.to(self.config.device)
        self.masked_visual_samples = self.mask_applicator(self.visual_samples, self.mask_test)

    def visual_sample_process(self):
        with torch.no_grad():
            self.visual_embeddings = self.feature_fusion(self.encoder(self.input_embedding(self.visual_samples)))
            self.visual_embeddings_sub = self.subspace_1(self.visual_embeddings)
            self.visual_masked_embeddings = []
            self.visual_masked_embeddings_sub = []
            for i in range(10):
                embeddings = self.feature_fusion(self.encoder(self.input_embedding(self.masked_visual_samples[:, i])))
                self.visual_masked_embeddings.append(embeddings)
                self.visual_masked_embeddings_sub.append(self.subspace_1(embeddings))
            self.visual_masked_embeddings = torch.stack(self.visual_masked_embeddings, dim=1)
            self.visual_masked_embeddings_sub = torch.stack(self.visual_masked_embeddings_sub, dim=1)

            # processing masks
            if self.config.mask_embedding:
                masks = 1 - self.mask_test
                k = torch.sum(masks, dim=-1, keepdim=True)
                mask_embed = (masks @ self.reduce_prompt_encoder) / (
                        torch.sum(masks, dim=-1, keepdim=True) ** 0.5 + 1e-6
                )
            else:
                if self.config.dense_mask:
                    masks = 2 * self.mask_test - 1
                    mask_embed = self.reduce_prompt_encoder(masks)  # (B, num, D), all index are transposed.
                else:
                    masks = 1 - self.mask_test
                    mask_embed = self.reduce_prompt_encoder(masks)
            self.mask_embeddings = mask_embed

            # prediction process
            self.pred_visual_masked_embeddings = []
            for i in range(10):
                pred = self.reduce_feature_predictor((self.visual_embeddings_sub + self.mask_embeddings[:, i, :]))
                self.pred_visual_masked_embeddings.append(pred)

            self.pred_visual_masked_embeddings = torch.stack(self.pred_visual_masked_embeddings, dim=1)

    def visual_sample_plot(self, index, masked_index, plot_num=1):
        original_sample = self.visual_samples[index].cpu()
        original_embed = self.visual_embeddings[index:index + 1].cpu()
        original_embed_sub = self.visual_embeddings_sub[index:index + 1].cpu()
        masked_samples = self.masked_visual_samples[index].cpu()
        masked_embeds = self.visual_masked_embeddings[index].cpu()
        masked_embed_sub = self.visual_masked_embeddings_sub[index].cpu()
        pred_masked_embeds = self.pred_visual_masked_embeddings[index].cpu()
        masks = self.mask_test[index].cpu()
        masks_embed = self.mask_embeddings[index].cpu()
        data = {
            "original_sample": self.visual_samples.cpu(),
            "original_embed": self.visual_embeddings.cpu(),
            "original_embed_sub": self.visual_embeddings_sub.cpu(),
            "masked_samples": self.masked_visual_samples.cpu(),
            "masked_embed": self.visual_masked_embeddings.cpu(),
            "masked_embed_sub": self.visual_masked_embeddings_sub.cpu(),
            "pred_masked_embeds": self.pred_visual_masked_embeddings.cpu(),
            "masks": self.mask_test.cpu(),
            "masks_embed": self.mask_embeddings.cpu()
        }
        # torch.save(data, os.path.join(self.model_path, "visual_data_{}.pt".format(plot_num)))
        plt.figure(figsize=(9, 9), dpi=300)
        plt.subplot(3, 3, 1)
        plt.plot(original_sample[:, 0])
        plt.title("Original Sample")
        plt.subplot(3, 3, 2)
        plt.imshow(original_embed, aspect="auto")
        plt.colorbar()
        plt.title("Original Embed")
        plt.subplot(3, 3, 3)
        plt.imshow(original_embed_sub, aspect="auto")
        plt.colorbar()
        plt.title("Original Embed-Sub")

        plt.subplot(3, 3, 4)
        plt.plot(masked_samples[masked_index])
        plt.title("Masked Sample: {}".format(masked_index))
        plt.subplot(3, 3, 5)
        plt.imshow(masked_embeds[masked_index:masked_index + 1], aspect="auto")
        plt.colorbar()
        plt.title("Masked Embed: {}".format(masked_index))
        plt.subplot(3, 3, 6)
        plt.imshow(masked_embed_sub[masked_index:masked_index + 1], aspect="auto")
        plt.colorbar()
        plt.title("Masked Embed-Sub: {}".format(masked_index))

        plt.subplot(3, 3, 7)
        plt.plot(masks[masked_index])
        plt.title("Mask".format(masked_index))
        plt.subplot(3, 3, 8)
        plt.imshow(pred_masked_embeds[masked_index:masked_index + 1], aspect="auto")
        plt.colorbar()
        plt.title("Pred Masked Embed: {}".format(masked_index))
        plt.subplot(3, 3, 9)
        plt.imshow(masks_embed[masked_index:masked_index + 1], aspect="auto")
        plt.colorbar()
        plt.title("Mask Embed: {}".format(masked_index))

        plt.savefig(os.path.join(self.model_path, "visual_{}.png".format(plot_num)))
        plt.cla()
        plt.close()


if __name__ == '__main__':
    config = PretrainConfig()
    config.pretrain_encoder = "Transformer"
    config.device = "cpu"
    embedding = DefaultInputEmbedding(config)
    inp = torch.randn((2, 30, 1))
    out = embedding(inp)
