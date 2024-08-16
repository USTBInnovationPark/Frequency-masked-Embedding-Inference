import os
import torch
import torch.nn as nn
import math

from config.configs import TimeDRLConfig
from models.pretrain_models import PretrainModel, ResNet
from models.RevIN import RevIN


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
                torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class PositionalEmbedding_trainable(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create a parameter tensor of size [max_length, d_model]
        pe = torch.randn(max_len, d_model).float()

        # Register it as a parameter that will be updated during training
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        # Just return the first T position embeddings
        return self.pe[None, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, last_dim, d_model, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2  # `same` padding
        self.tokenConv = nn.Conv1d(
            in_channels=last_dim,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        # x.shape = (Batch, patch_num, patch_len)
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(-1, -2)
        return x


class DataEmbedding(nn.Module):
    def __init__(
            self,
            patch_size,  # last dimension of input
            patch_step,
            d_model,
            dropout=0.1,
            pos_embed_type="none",
            token_embed_type="linear",
            token_embed_kernel_size=3,
            device="cuda"
    ):
        super().__init__()

        # Positional embedding (none, learnable, fixed)
        if pos_embed_type == "none":
            self.position_embedding = None
        elif pos_embed_type == "learnable":  # nn.Parameter
            self.position_embedding = PositionalEmbedding_trainable(d_model)
        elif pos_embed_type == "fixed":  # sin/cos
            self.position_embedding = PositionalEmbedding(d_model)
        else:
            raise NotImplementedError

        # Token embedding (linear, conv)
        if token_embed_type == "linear":
            self.token_embedding = nn.Linear(patch_size, d_model, bias=False)
        elif token_embed_type == "conv":
            self.token_embedding = TokenEmbedding(
                last_dim=patch_size, d_model=d_model, kernel_size=token_embed_kernel_size
            )
        else:
            raise NotImplementedError
        self.patch_size = patch_size
        self.patch_step = patch_step
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, patch_size, device=device)
        )
        self.revIn = RevIN(1, affine=False)

        # Dropout
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):  # (B, T, C)
        B = x.shape[0]
        # x = self.revIn(x, "norm")
        x_p = self.patch(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x_p = torch.cat([cls_token, x_p], dim=1)  # (batch*channel, patch_num + 1, patch_size)
        # Token embedding
        x_p = self.token_embedding(x_p)  # (B, T, D)

        # Position embedding
        if self.position_embedding is not None:
            x_p = x_p + self.position_embedding(x_p)  # (B, T, D)

        return self.dropout(x_p)

    def patch(self, x):  # x.shape = (batch, channel, length, )
        x = x.transpose(-1, -2)  # x.shape = (batch, channel, length)
        x = x.unfold(dimension=-1,
                     size=self.patch_size,
                     step=self.patch_step).contiguous()  # x.shape = (batch, channel, patch_num, patch_size)
        x = x.view(-1, x.shape[-2], x.shape[-1])  # x.shape = (batch*channel, patch_num, patch_size)
        return x


class TimeDRL(PretrainModel):
    def __init__(self, config: TimeDRLConfig):
        super(TimeDRL, self).__init__(config)
        if config.pretrain_encoder == "ResNet":
            self.input_embedding = DataEmbedding(config.patch_size,
                                                 config.patch_step,
                                                 16,
                                                 config.dropout,
                                                 config.pos_embed_type,
                                                 config.embed_type,
                                                 3,
                                                 config.device)
            # set stride=1, make sure the output length unchanged
            self.encoder = ResNet(config.encoder_size, 1)
        else:
            self.input_embedding = DataEmbedding(config.patch_size,
                                                 config.patch_step,
                                                 config.hidden_dim,
                                                 config.dropout,
                                                 config.pos_embed_type,
                                                 config.embed_type,
                                                 3,
                                                 config.device)
        self.predictive_linear = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim, config.patch_size),
        )
        self.contrastive_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
        )
        # losses
        self.mse = nn.MSELoss()
        self.cos_sim = nn.CosineSimilarity(dim=-1)
        self.config = config
        self.to(config.device)

    def forward(self, x):
        x_p_1 = self.input_embedding.patch(x)
        x_p_2 = self.input_embedding.patch(x)
        x_1 = self.input_embedding(x)
        x_2 = self.input_embedding(x)
        # print(p_1.shape)
        z_1 = self.encoder(x_1)
        z_2 = self.encoder(x_2)
        x_1_pred = self.predictive_linear(z_1[:, 1:, :])
        x_2_pred = self.predictive_linear(z_2[:, 1:, :])
        i_1 = z_1[:, 0, :]
        i_2 = z_2[:, 0, :]
        # print(i_1.shape)
        i_1_pred = self.contrastive_predictor(i_1)
        i_2_pred = self.contrastive_predictor(i_2)
        return x_p_1, x_p_2, x_1_pred, x_2_pred, i_1, i_2, i_1_pred, i_2_pred

    def compute_loss(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     criterion) -> torch.Tensor:
        x_1, x_2, x_1_pred, x_2_pred, i_1, i_2, i_1_pred, i_2_pred = self(x)
        contrastive_loss = -(self.cos_sim(i_1.detach(), i_2_pred).mean() +
                             self.cos_sim(i_2.detach(), i_1_pred).mean()) * 0.5
        predictive_loss = (self.mse(x_1_pred, x_1) +
                           self.mse(x_2_pred, x_2)) * 0.5
        return self.config.contrastive_weight * contrastive_loss + predictive_loss


if __name__ == '__main__':
    config = TimeDRLConfig()
    config.pretrain_encoder = "Transformer"
    config.device = "cpu"
    net = TimeDRL(config)
    inp = torch.randn((2, 178, 1))
    _ = net.compute_loss(inp, inp, None)
