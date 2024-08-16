from util.trainer import *
from config.configs import *
import torch
import numpy as np
import argparse


def get_pretrain_config(target_model="FMEI"):
    if target_model == "FEI":
        config = FreqMaskJEConfig()

    elif target_model == "SimMTM":
        config = SimMTMConfig()

    elif target_model == "TimeDRL":
        config = TimeDRLConfig()

    elif target_model == "InfoTS":
        config = InfoTSConfig()
    else:
        raise ValueError("Unknown target method:{}".format(target_model))
    return config


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    return seed


if __name__ == '__main__':
    set_seed(2024)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_type', default='p', type=str,
                        help='\'p\'(pre-training) or \'f\'(fine-tuning) or \'l\'(linear evaluation)')
    parser.add_argument('--task', default='c', type=str,
                        help='\'c\'(classification) or \'r\'(regression). '
                             'This is only used when \'--task=f\' or \'--task=l\'.')
    parser.add_argument('--model', default=None, type=str,
                        help='The path to the pre-trained model. This is only used when \'--task=f\' or \'--task=l\'. '
                             'If not provided, the model will be fine-tuned with random initialization.')
    parser.add_argument('--dataset', default='SLE', type=str,
                        help='Pretrain dataset: SLE (SleepEEG).\n '
                             'Classification dataset: GES/FDB/EMG/EPI/HAR/UCR.\n'
                             'Regression dataset: FD001/FD002/FD003/FD004/OPA/OPB/OPC')
    parser.add_argument('--method', default='FEI', type=str,
                        help='\'FEI\'/\'SimMTM\'/\'TimeDRL\'/\'InfoTS\'. This is only used when \'--task=p\'. ')
    parser.add_argument('--device', default='cuda:0', type=str,
                        help='\'cuda:n\'/\'cpu\'')
    args, unknown = parser.parse_known_args()
    task_type = args.task_type
    task = args.task
    model = args.model
    dataset = args.dataset
    method = args.method
    device = args.device
    pretrain_config = get_pretrain_config(method)
    pretrain_config.device = device

    if task_type == 'p':
        target = method
        pretrain_config.pretrain_dataset = dataset
        model = pre_train(pretrain_config, target)
    elif task_type == 'l':
        if task == 'c':
            config = DownstreamConfig_cls()
            config.finetune_dataset = dataset
            config.finetune_encoder = False
            config.device = device
            if dataset == "ucr":
                fine_tune_UCR(pretrain_config, config, model, method)
            else:
                fine_tune_cls(pretrain_config, config, model, method)
        elif task == 'r':
            config = DownstreamConfig_reg()
            config.finetune_dataset = args.dataset
            config.finetune_encoder = False
            config.device = device
            fine_tune_reg(pretrain_config, config, model, method)
        else:
            raise ValueError("Unknown task: {}".format(task))
    elif task_type == 'f':
        if task == 'c':
            config = DownstreamConfig_cls()
            config.finetune_dataset = dataset
            config.finetune_encoder = True
            config.finetune_epoch = 100
            config.finetune_lr = 1e-5
            config.device = device
            fine_tune_cls(pretrain_config, config, model, method)
        elif task == 'r':
            config = DownstreamConfig_reg()
            config.finetune_dataset = args.dataset
            config.finetune_encoder = True
            config.finetune_epoch = 100
            config.finetune_lr = 1e-5
            config.device = device
            fine_tune_reg(pretrain_config, config, model, method)
        else:
            raise ValueError("Unknown task: {}".format(task))
    else:
        raise ValueError("Unknown task type: {}".format(task_type))
