from train.config import BaseConfig


class PretrainConfig(BaseConfig):
    def __init__(self):
        super(PretrainConfig, self).__init__()
        # ------------- Pre-train Configs ------------- #
        self.pretrain_sample_length = 178  # sample length used in pre-training
        self.pretrain_dataset = "SLE"  # available values are listed in DatasetEnum
        self.pretrain_batch_size = 512  # batch length used in pre-training
        self.pretrain_num_workers = 0  # number of thread workers used in pre-training
        self.pretrain_epoch = 100  # maximum epoch
        self.pretrain_lr = 0.0002  # initial learning rate
        self.pretrain_sch = "step"  # 'step'/'warm'. Indicating the lr scheduler. Where 'step' is ExponentialLR
        self.pretrain_lr_gamma = 0.90  # The value of decay gamma. Only used when sch='step'.
        self.feature_fusion = "mean"  # How to fusion the temporal dimension of original feature. first/mean/all/last
        self.encoder_size = "big"  # The size of 1D ResNet. 'tiny'/'small'/'normal'/'big'
        self.pretrain_early_stop = 5  # Patience value of early stopping in pre-training
        self.dropout = 0  # Dropout is not used in FEI encoder.
        self.amp = False  # Weather to use Automatic Mixed Precision (AMP). It is useful when your RAM is small.
        self.device = "cuda:1"  # Which device to run.
        # ------------- ResNet-Only ------------- #
        # The stride size used in 1D ResNet is recommended to be set to 1 or 2. A stride of 1 will produce temporal
        # invariance features. This setting does not significantly impact FEI, but some baselines require temporal
        # invariance features.
        self.stride = 2


class FEIConfig(PretrainConfig):
    def __init__(self):
        super(FEIConfig, self).__init__()
        self.mask_pool_size = 1000
        self.reduce_mask_ratio = [0., 0.7]
        self.mask_aug_during_training = False
        self.target_num = 1
        self.dense_mask = False
        self.mask_embedding = True
        self.exponential_moving_average_rate = 0.995
        self.visual_samples = False
        self.mask_type = "discrete"  # "continue"/"temporal"


class SimMTMConfig(PretrainConfig):
    def __init__(self):
        super(SimMTMConfig, self).__init__()
        self.mask_ratio = 0.5
        self.lm = 3
        self.positive_nums = 3
        self.temperature = 0.2
        self.pretrain_batch_size = 32
        self.pretrain_epoch = 30


class TimeDRLConfig(PretrainConfig):
    def __init__(self):
        super(TimeDRLConfig, self).__init__()
        self.embed_type = "conv"  # linear/conv
        self.pos_embed_type = "learnable"  # learnable/fixed/none
        self.dropout = 0.1
        self.feature_fusion = "first"  # This is not used in Pretrain phase
        self.contrastive_weight = 0.1
        self.patch_size = 16
        self.patch_step = 8
        self.pretrain_batch_size = 32
        self.pretrain_epoch = 30
        self.pretrain_early_stop = 0


class InfoTSConfig(PretrainConfig):
    def __init__(self):
        super(InfoTSConfig, self).__init__()
        self.pretrain_lr = 0.001
        self.meta_lr = 0.01
        self.aug_p1 = 0.2
        self.aug_p2 = 0.0
        self.used_augs = None
        self.mask_mode = 'binomial'
        self.dropout = 0.1
        self.split_number = 8
        self.meta_beta = 0.1
        self.beta = 0.5
        self.stride = 1
        self.pretrain_early_stop = 0


class FineTuneConfig(BaseConfig):
    def __init__(self):
        super(FineTuneConfig, self).__init__()
        # ------------- Fine-tune Configs ------------- #
        self.finetune_dataset = "FDB"  # available values are listed in DatasetEnum
        self.finetune_sample_length = 178
        self.finetune_batch_size = 32
        self.finetune_num_workers = 0
        self.finetune_epoch = 300
        self.finetune_lr = 0.0001
        self.finetune_lr_sch_step = 2
        self.finetune_lr_gamma = 0.7
        self.finetune_encoder = False
        self.affine_bn = False
        self.amp = False
        self.device = "cuda:1"


class DownstreamConfig_cls(FineTuneConfig):
    def __init__(self):
        super(DownstreamConfig_cls, self).__init__()
        self.finetune_dataset = "FDB"  # available values are listed in DatasetEnum
        self.cls_num = 3


class DownstreamConfig_pred(FineTuneConfig):
    def __init__(self):
        super(DownstreamConfig_pred, self).__init__()
        self.finetune_dataset = "WEA"  # available values are listed in DatasetEnum
        self.pred_len = 96
        self.train_data = 0.1
        self.affine_bn = False


class DownstreamConfig_reg(FineTuneConfig):
    def __init__(self):
        super(DownstreamConfig_reg, self).__init__()
        self.finetune_dataset = "FD001"
