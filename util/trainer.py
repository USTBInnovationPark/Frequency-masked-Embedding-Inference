import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import datasets_clsa.DataGenerator as cls_data
import datasets_clsa.DatasetEnum as cls_util
import datasets_reg.cmapss as reg_data_cmapss
import datasets_reg.bearing as reg_data_xjtu
import os
from config.configs import PretrainConfig, FineTuneConfig

from models.pretrain_models import FreqMaskJECNNModel, PretrainModel, ClassifierModel, RegressionModel
from models.SimMTM import SimMTM
from models.TimeDRL import TimeDRL
from models.InfoTS import InfoTS
from train.config import build_flag
from util.scheduler import WarmupCosineSchedule

AVAILABLE_TARGETS = ["FEI", "SimMTM", "TimeDRL", "InfoTS"]


def build_pretrain_model(config: PretrainConfig,
                         target):
    assert target in AVAILABLE_TARGETS
    if target == AVAILABLE_TARGETS[0]:
        config.model_flag = build_flag("FEI",
                                       preTrain=config.pretrain_dataset)
        model = FreqMaskJECNNModel(config)
    elif target == AVAILABLE_TARGETS[1]:
        config.model_flag = build_flag("SimMTM",
                                       preTrain=config.pretrain_dataset)
        model = SimMTM(config)
    elif target == AVAILABLE_TARGETS[2]:
        config.model_flag = build_flag("TimeDRL",
                                       preTrain=config.pretrain_dataset)
        model = TimeDRL(config)
    elif target == AVAILABLE_TARGETS[3]:
        config.model_flag = build_flag("InfoTS",
                                       preTrain=config.pretrain_dataset)
        model = InfoTS(config)
    else:
        raise NotImplementedError("Unknown target pretrain method {}.".format(target))
    return model


def build_fine_tune(pretrain_config: PretrainConfig,
                    fine_tune_config: FineTuneConfig,
                    model: PretrainModel or str,
                    finetune_model_type,
                    target,
                    cls_num=-1):
    if isinstance(fine_tune_config, str):
        fine_tune_config = FineTuneConfig().load(fine_tune_config)
    if isinstance(pretrain_config, str):
        pretrain_config = PretrainConfig().load(pretrain_config)
    if cls_num != -1:
        fine_tune_config.add_param("cls_num", int(cls_num))
    fine_tune_config.model_flag = build_flag("Finetune",
                                             m=target if model is not None else "None",
                                             pretrain=pretrain_config.pretrain_dataset if model is not None else "None",
                                             fineTune=fine_tune_config.finetune_dataset)
    init_model = build_pretrain_model(pretrain_config, target)
    finetune_model = finetune_model_type(fine_tune_config, init_model)
    if model is not None:
        finetune_model.logger.info("Loaded model pretrained on {}".format(pretrain_config.pretrain_dataset))
        finetune_model.load_pretrain_model(model)
    else:
        finetune_model.logger.info("Finetuning without pretrained model!")
    return pretrain_config, fine_tune_config, finetune_model


def pre_train(config, target_model="FEI"):
    model = build_pretrain_model(config, target_model)
    pre_train_data = cls_data.DefaultGenerator(cls_data.DatasetName.__members__[config.pretrain_dataset],
                                               flag='train',
                                               x_len=config.pretrain_sample_length)
    val_data = cls_data.DefaultGenerator(cls_data.DatasetName.__members__[config.pretrain_dataset],
                                         flag='val',
                                         x_len=config.pretrain_sample_length)
    model.prepare_data(pre_train_data,
                       eval_set=val_data,
                       eval_shuffle=True,
                       batch_size=config.pretrain_batch_size,
                       num_workers=config.pretrain_num_workers)
    if target_model == AVAILABLE_TARGETS[3]:
        # The parameters in the InfoTS is not all updated in general training phase.
        # Some of params will be updated with customized process in InfoTS.epoch_start(..).
        params = list(model.input_embedding.parameters()) + list(model.encoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=config.pretrain_lr)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=config.pretrain_lr, )
    if config.pretrain_sch == "step":
        # lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
        #                                                     T_max=config.pretrain_epoch)
        lr_sch = torch.optim.lr_scheduler.ExponentialLR(optimizer, config.pretrain_lr_gamma)
    else:
        lr_sch = WarmupCosineSchedule(optimizer=optimizer,
                                      warmup_steps=5,
                                      start_lr=1e-6,
                                      ref_lr=config.pretrain_lr,
                                      T_max=config.pretrain_epoch // 2,
                                      final_lr=1e-7)
    model.train_model(epoch=config.pretrain_epoch,
                      criterion=nn.MSELoss(),
                      optimizer=optimizer,
                      lr_schedular=lr_sch,
                      early_stop=config.pretrain_early_stop,
                      amp=config.amp,
                      auto_test=False)
    return model


def fine_tune_cls(pretrain_config: PretrainConfig or str,
                  fine_tune_config: FineTuneConfig or str,
                  model: PretrainModel or str or None,
                  target):
    fine_tune_config.cls_num = cls_util.get_cls_num(fine_tune_config.finetune_dataset)
    _, fine_tune_config, finetune_model = build_fine_tune(pretrain_config,
                                                          fine_tune_config,
                                                          model,
                                                          ClassifierModel,
                                                          target)
    fintune_train_data = cls_data.DefaultGenerator(cls_data.DatasetName.__members__[fine_tune_config.finetune_dataset],
                                                   flag='train',
                                                   x_len=fine_tune_config.finetune_sample_length)
    fintune_val_data = cls_data.DefaultGenerator(cls_data.DatasetName.__members__[fine_tune_config.finetune_dataset],
                                                 flag='val',
                                                 x_len=fine_tune_config.finetune_sample_length)
    fintune_test_data = cls_data.DefaultGenerator(cls_data.DatasetName.__members__[fine_tune_config.finetune_dataset],
                                                  flag='test',
                                                  x_len=fine_tune_config.finetune_sample_length)
    finetune_model.prepare_data(fintune_train_data, fintune_test_data, fintune_val_data,
                                fine_tune_config.finetune_batch_size,
                                fine_tune_config.finetune_num_workers)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, finetune_model.parameters()),
                                  lr=fine_tune_config.finetune_lr)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                        T_max=fine_tune_config.finetune_epoch)
    finetune_model.train_model(epoch=fine_tune_config.finetune_epoch,
                               criterion=torch.nn.CrossEntropyLoss(),
                               optimizer=optimizer,
                               lr_schedular=lr_sch,
                               early_stop=15,
                               amp=fine_tune_config.amp,
                               auto_test=True)
    return finetune_model, finetune_model.test_results


def fine_tune_UCR(pretrain_config: PretrainConfig or str,
                  fine_tune_config: FineTuneConfig or str,
                  model: PretrainModel or str or None,
                  target):
    datafiles = os.listdir("./UCR")
    datafiles.sort()
    if target == "TimeDRL":
        datafiles.remove("SmoothSubspace")
    print(datafiles)
    test_results = []
    mean_acc = 0
    for data in datafiles:
        train_data, test_data, cls_num = cls_data.load_UCR(data)
        _, fine_tune_config, finetune_model = build_fine_tune(pretrain_config,
                                                              fine_tune_config,
                                                              model,
                                                              ClassifierModel,
                                                              target,
                                                              cls_num=cls_num)
        finetune_model.prepare_data(train_data, test_data, None,
                                    fine_tune_config.finetune_batch_size,
                                    fine_tune_config.finetune_num_workers)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, finetune_model.parameters()),
                                      lr=fine_tune_config.finetune_lr)
        lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                            T_max=fine_tune_config.finetune_epoch)
        finetune_model.train_model(epoch=fine_tune_config.finetune_epoch,
                                   criterion=torch.nn.CrossEntropyLoss(),
                                   optimizer=optimizer,
                                   lr_schedular=lr_sch,
                                   early_stop=0,
                                   amp=fine_tune_config.amp,
                                   auto_test=True)
        test_results.append(finetune_model.test_results)
        mean_acc += finetune_model.test_results["acc"]
    torch.save(test_results, finetune_model.model_path+"UCR_results.pt")
    print("Mean Acc: {}".format(mean_acc / len(datafiles)))
    return finetune_model, test_results, datafiles


def fine_tune_reg(pretrain_config: PretrainConfig or str,
                  fine_tune_config: FineTuneConfig or str,
                  model: PretrainModel or str or None,
                  target):
    _, fine_tune_config, finetune_model = build_fine_tune(pretrain_config,
                                                          fine_tune_config,
                                                          model,
                                                          RegressionModel,
                                                          target)
    # Get dataset
    if fine_tune_config.finetune_dataset in reg_data_xjtu.Condition.__members__:
        train_data = reg_data_xjtu.XJTU(reg_data_xjtu.DEFAULT_ROOT,
                                        [reg_data_xjtu.Condition.__members__[
                                             fine_tune_config.finetune_dataset]],
                                        [[1]],
                                        [[1]],
                                        [[-1]],
                                        reg_data_xjtu.LabelsType.TYPE_P,
                                        window_size=8192,
                                        step_size=81920)
        val_data = reg_data_xjtu.XJTU(reg_data_xjtu.DEFAULT_ROOT,
                                      [reg_data_xjtu.Condition.__members__[
                                           fine_tune_config.finetune_dataset]],
                                      [[2]],
                                      [[1]],
                                      [[-1]],
                                      reg_data_xjtu.LabelsType.TYPE_P,
                                      window_size=8192,
                                      step_size=8192)
        test_data = reg_data_xjtu.XJTU(reg_data_xjtu.DEFAULT_ROOT,
                                       [reg_data_xjtu.Condition.__members__[
                                            fine_tune_config.finetune_dataset]],
                                       [[3, 4, 5]],
                                       [[1, 1, 1]],
                                       [[-1, -1, -1]],
                                       reg_data_xjtu.LabelsType.TYPE_P,
                                       window_size=8192,
                                       step_size=8192)
        train_data.raw_data = train_data.raw_data[:, :1]  # Only one variable is considered
        val_data.raw_data = val_data.raw_data[:, :1]  # Only one variable is considered
        test_data.raw_data = test_data.raw_data[:, :1]  # Only one variable is considered
        scaler = reg_data_xjtu.XJTUScaler()
        scaler.fit_transform(train_data)
        scaler.transform(val_data)
        scaler.transform(test_data)
    elif fine_tune_config.finetune_dataset in reg_data_cmapss.Subset.__members__:
        cmapss = reg_data_cmapss.get_data(reg_data_cmapss.DEFAULT_ROOT,
                                          reg_data_cmapss.Subset.__members__[fine_tune_config.finetune_dataset],
                                          35, 1,
                                          ["s_2"],
                                          rul_threshold=125,
                                          label_norm=True,
                                          val_ratio=0.2)
        train_data = cmapss[0]
        val_data = cmapss[2]
        test_data = cmapss[1]
    else:
        raise ValueError(f"Unknown regression dataset: {fine_tune_config.finetune_dataset}.")
    finetune_model.prepare_data(train_data, test_data, val_data,
                                fine_tune_config.finetune_batch_size,
                                fine_tune_config.finetune_num_workers)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, finetune_model.parameters()),
                                  lr=fine_tune_config.finetune_lr)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                        T_max=fine_tune_config.finetune_epoch)
    finetune_model.train_model(epoch=fine_tune_config.finetune_epoch,
                               criterion=torch.nn.MSELoss(),
                               optimizer=optimizer,
                               lr_schedular=lr_sch,
                               early_stop=0,
                               amp=fine_tune_config.amp,
                               auto_test=True)
    return finetune_model
