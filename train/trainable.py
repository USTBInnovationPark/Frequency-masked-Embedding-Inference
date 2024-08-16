from typing import Union, Callable, Iterable
from torch.utils.data import Dataset, DataLoader
from train.config import BaseConfig
from os.path import join

import os
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import logging

import time

root = os.path.dirname(__file__)


def _check_path(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


class TrainableModule(nn.Module):
    """
    The base module of Trainable Models. So call 'trainable' means the models can be trained by
    following method easily:

        >>> model.prepare_data(...)
        >>> model.train_model(...)
    """

    def __init__(self, base_config: BaseConfig):
        super(TrainableModule, self).__init__()
        self.eval_losses = None
        self.train_losses = None

        self.eval_loader = None
        self.test_loader = None
        self.train_loader = None

        self.optimizer = None
        self.criterion = None
        self.lr_scheduler = None

        self.opt_model = None

        self.config = base_config
        self.flag = base_config.model_flag
        self.device = base_config.device

        self.__model_path = None
        self.__logger = None

    def prepare_data(self,
                     train_set: Dataset,
                     test_set: Dataset = None,
                     eval_set: Dataset = None,
                     batch_size: int = 256,
                     num_workers: int = 8,
                     collator_fn=None,
                     eval_shuffle=True):
        self.train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
                                       collate_fn=collator_fn)
        self.logger.info("train size:{}".format(len(train_set)))
        if test_set is not None:
            self.test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=num_workers)
            self.logger.info("test size:{}".format(len(test_set)))
        if eval_set is not None:
            self.eval_loader = DataLoader(eval_set, batch_size, shuffle=eval_shuffle, num_workers=num_workers)
            self.logger.info("validate size:{}".format(len(eval_set)))

    def _train_init(self,
                    optimizer: Union[torch.optim.Optimizer, str, Iterable] = "adam",
                    lr: Union[float, None] = None,
                    lr_schedular: Union[torch.optim.lr_scheduler.LRScheduler, Callable, str, Iterable, None] = None,):
        self.config.save(self.model_path)
        if self.train_loader is None:
            self.logger.error("The data_loader is None! Set the param data_loader not None or use "
                              "model.prepare_data(Dataset, batch_size, num_workers) to provide the"
                              "training data.")
            raise RuntimeError()
        if isinstance(optimizer, str):
            if lr is None:
                self.logger.error("The argument 'lr' can not be None if optimizer is given by name."
                                  "The None type can be available only when the optimizer is given"
                                  "a torch.optim.Optimizer instance.")
                raise RuntimeError()
            if optimizer == "adam":
                self.optimizer = torch.optim.Adam(lr=lr, params=self.parameters())
            elif optimizer == "rms":
                self.optimizer = torch.optim.RMSprop(lr=lr, params=self.parameters())
            elif optimizer == "sgd":
                self.optimizer = torch.optim.SGD(lr=lr, params=self.parameters())
            else:
                self.logger.error("Unknown optimizer {}. The optimizer should be ['adam', 'rms', 'sgd'],"
                                  "or given a torch.optim.Optimizer instance.".format(optimizer))
                raise RuntimeError()
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        if lr_schedular is not None:
            if isinstance(lr_schedular, str):
                assert lr_schedular in ["cosine", "warm", "step"]
                if lr_schedular == "cosine":
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                   T_max=5,
                                                                                   eta_min=1e-5)
                elif lr_schedular == "warm":
                    self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                                             T_0=5,
                                                                                             T_mult=2,
                                                                                             eta_min=1e-5)
                elif lr_schedular == "step":
                    self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)
                else:
                    self.logger.error("Unknown scheduler {}. The scheduler should be ['cosine', 'warm', 'step'],"
                                      "or given a Scheduler instance."
                                      .format(lr_schedular))
                    raise RuntimeError()

            elif isinstance(lr_schedular, torch.optim.lr_scheduler.LRScheduler) or isinstance(lr_schedular, Iterable):
                self.lr_scheduler = lr_schedular
            elif isinstance(lr_schedular, Callable):
                self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_schedular, last_epoch=-1)
            else:
                self.lr_scheduler = lr_schedular

    def train_model(self,
                    epoch: int,
                    criterion: nn.Module,
                    optimizer: Union[torch.optim.Optimizer, str] = "adam",
                    lr: Union[float, None] = None,
                    lr_schedular: Union[torch.optim.lr_scheduler.LRScheduler, Callable, str, None] = None,
                    early_stop: Union[int, None] = 0,
                    amp=False,
                    show_batch_loss=False,
                    auto_test=True):
        """
        Call this method to train the Trainable Modules.

        :param epoch: Training Epoch.
        :param lr: The initial Learning Rate. It will be ignored when the optimizer is given by an Optimizer instance.
        :param criterion: The Loss Function.
        :param optimizer: Should be one of ['adam', 'rms', 'sgd'] or a instance of torch.optim.Optimizer. If given
                          the first format, the lr must be indicated. else the lr will be ignored. Default 'adam'.
        :param lr_schedular: The learning rate reduce schedular. A lambda function (epoch), one of ['cosine', 'warm',
                             'step'] or an instance of torch.optim.lr_scheduler.LRScheduler. Default None.
        :param early_stop: Early stopping patience. If 0, early stopping is not applied. Default 0.
        :param amp: Weather using Automatic Mixed Precision. Default False.
        :param show_batch_loss: Weather showing the loss information in every batch. Default False.
        :param auto_test: Weather testing model when finishing training. Default True.
        """
        self._train_init(optimizer, lr, lr_schedular)
        if early_stop is not None and early_stop > 0:
            mini_eval_loss = None
            patience = early_stop
            now_patience = 0
        if amp:
            from torch.cuda.amp import GradScaler, autocast
            grad_scaler = GradScaler()
        self.criterion = criterion
        self.train_losses = []
        self.eval_losses = []
        self.logger.info("Model flag: {}".format(self.flag))
        self.logger.info("Start training epoch {}".format(epoch))
        import torch.profiler

        # training
        start_time = time.time()
        self.train_start()  # callback function
        for e in range(epoch):
            self.epoch_start(e)  # callback function
            self.logger.info("epoch: {}/{}".format(e + 1, epoch))
            epoch_start_time = time.time()
            batch_losses = []
            for step, (x, y) in enumerate(self.train_loader):
                self.iter_start(step)
                if amp:
                    with autocast():
                        x = x.to(torch.float32).to(self.device)
                        y = y.to(torch.float32).to(self.device)
                        loss = self.compute_loss(x, y, self.criterion)
                else:
                    x = x.to(torch.float32).to(self.device)
                    y = y.to(torch.float32).to(self.device)
                    loss = self.compute_loss(x, y, self.criterion)
                if show_batch_loss:
                    self.logger.info(
                        "\tbatch: {}/{}, loss:{:.4f}".format(step + 1, len(self.train_loader), loss.item()))

                self.optimizer.zero_grad()
                if amp:
                    grad_scaler.scale(loss).backward()
                    self.iter_end_before_opt(step)
                    grad_scaler.step(self.optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    self.iter_end_before_opt(step)
                    self.optimizer.step()
                batch_losses.append(loss.item())
                self.iter_end(step)

            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, Iterable):
                    for scheduler in self.lr_scheduler:
                        print(1)
                        scheduler.step()
                else:
                    self.lr_scheduler.step()

            batch_loss = np.average(batch_losses)
            self.train_losses.append(batch_loss)
            batch_losses.clear()
            # if e % 5 == 0:
            #     self.logger.info("\tSaving check point...")
            #     torch.save(self.state_dict(), join(self.model_path, 'check_point_{}.pt'.format(e)))

            # evaluation
            if self.eval_loader is not None:
                self.eval()
                eval_losses = []

                with torch.no_grad():
                    for step, (e_x, e_y) in enumerate(self.eval_loader):
                        e_x = e_x.to(torch.float32).to(self.device)
                        e_y = e_y.to(torch.float32).to(self.device)
                        loss = self.compute_loss(e_x, e_y, self.criterion)
                        eval_losses.append(loss.item())
                    eval_loss = np.average(eval_losses)
                    self.eval_losses.append(eval_loss)
                self.logger.info("\ttraining loss: {:.4}\n \teval loss: {:.4} \tCurrent learning rate: {}".
                                 format(batch_loss, eval_loss, self._get_current_learning_rate()))
            else:
                self.logger.info("\ttraining loss: {:.4}\n \tCurrent learning rate: {}".
                                 format(batch_loss, self._get_current_learning_rate()))
            self.logger.info("\tEpoch time spent: %s s" % (time.time() - epoch_start_time))
            self.epoch_end(e)  # callback function
            # early stop
            if early_stop > 0:
                if mini_eval_loss is None:
                    mini_eval_loss = eval_loss
                    torch.save(self.state_dict(), join(self.model_path, 'check_point.pt'))
                    continue
                if eval_loss >= mini_eval_loss:
                    now_patience = now_patience + 1
                    self.logger.info("\tEarly Stopping Monitor: bigger eval loss, now patience score {}/{}"
                                     .format(now_patience, patience))
                else:
                    now_patience = 0
                    mini_eval_loss = eval_loss
                    self.logger.info("\tEarly Stopping Monitor: smaller eval loss achieved, saving model...")
                    torch.save(self.state_dict(), join(self.model_path, 'check_point.pt'))
                if now_patience >= patience:
                    self.logger.info("\tEarly Stopping in epoch {}".format(e))
                    self.logger.info("=" * 20 + "Best Eval Loss: {:.4f}".format(mini_eval_loss) + "=" * 20)
                    self.load_state_dict(torch.load(join(self.model_path, 'check_point.pt')))
                    break
        end_time = time.time()
        self.train_end()  # callback function
        self.logger.info("Total time spent: %s s" % round(end_time - start_time, 2))
        self.plot_losses()
        torch.save(self.state_dict(), join(self.model_path, 'check_point.pt'))
        if auto_test:
            self.test_model()

    def _get_current_learning_rate(self):
        c_lr = ''
        c_lr += "{}\t".format(self.optimizer.state_dict()['param_groups'][0]['lr'])
        return c_lr

    def test_model(self):
        if self.test_loader is None:
            self.logger.warning("The test_loader is None! test_model() will not be processed.")
            return
        self.test_start()  # callback function
        output = None
        labels = None
        self.eval()
        with torch.no_grad():
            index = 1
            for step, (x, y) in enumerate(self.test_loader):
                x = x.to(torch.float32).to(self.device)
                y = y.to(torch.float32).to(self.device)
                model_out = self(x)
                model_out = model_out.detach().cpu()
                y = y.detach().cpu()
                output = torch.cat([output, model_out], dim=0) if output is not None else model_out
                labels = torch.cat([labels, y], dim=0) if labels is not None else y
                if output.numel() >= 40000000:  # result cut
                    np.save(join(self.model_path, "model_test_output_part{}".format(index)),
                            output.cpu().detach().numpy())
                    np.save(join(self.model_path, "model_test_labels_part{}".format(index)),
                            labels.cpu().detach().numpy())
                    output, labels = None, None
                    index += 1
            # output = torch.cat(output, dim=0)
            # labels = torch.cat(labels, dim=0)
            if output is not None:
                np.save(join(self.model_path, "model_test_output_part{}".format(index)),
                        output.cpu().detach().numpy())
                np.save(join(self.model_path, "model_test_labels_part{}".format(index)),
                        labels.cpu().detach().numpy())
        self.test_end()  # callback function

    def set_criterion(self,
                      criterion):
        self.criterion = criterion

    def plot_losses(self, show=False):
        if self.train_losses is None or self.eval_losses is None:
            self.logger.error("The model is not trained by internal training method. "
                              "You could call plot_losses(show=False) after training the model by:"
                              ">>> model.prepare_data(...)"
                              ">>> model.train_model(...)."
                              "Tips: plot_losses(show=False) will not work if you train your model manually"
                              "but not the above process.")
            return
        if show:
            matplotlib.use("QtAgg")
        else:
            matplotlib.use("Agg")
        plt.suptitle("Model Loss")
        plt.plot(self.train_losses, label="training loss")
        plt.plot(self.eval_losses, label="evalidate loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.model_path, "train_eval_losses.png"))
        plt.cla()
        if show:
            plt.show(block=True)

    @property
    def model_path(self):
        if self.__model_path is not None:
            _check_path(self.__model_path)
            return self.__model_path
        else:
            model_index = 1
            path = os.path.join(root, "model_result", self.flag)
            while os.path.exists(path + "_" + str(model_index)):
                model_index += 1
            self.__model_path = path + "_" + str(model_index)
            _check_path(self.__model_path)
            return self.__model_path

    @property
    def logger(self):
        if self.__logger is not None:
            return self.__logger
        else:
            log_file = join(self.model_path, "train.log")
            handler_test = logging.FileHandler(log_file)  # stdout to file
            handler_control = logging.StreamHandler()  # stdout to console
            handler_test.setLevel('INFO')
            handler_control.setLevel('INFO')
            fmt = '%(asctime)s - %(levelname)s - %(message)s'
            formatter = logging.Formatter(fmt)
            handler_test.setFormatter(formatter)
            self.__logger = logging.getLogger(str(self.__hash__()))
            self.__logger.setLevel(logging.INFO)
            self.__logger.addHandler(handler_test)
            self.__logger.addHandler(handler_control)
            return self.__logger

    def _criterion(self, y, label):
        if label.device.type == 'cpu':
            y = y.detach().cpu()
            label = label.detach().cpu()
        elif label.device.type != y.device.type:
            y = y.to(label.device)
        return self.criterion(y, label)

    def compute_loss(self,
                     x: torch.Tensor,
                     y: torch.Tensor,
                     criterion) -> torch.Tensor:
        """
        :param x: The input tensors.
        :param y: The label tensors.
        :param criterion: The loss function.

        An overridable method for different process of loss computation. The default process is simple
        single output computation. This method should only be overrideen if custom loss computation is
        required when training the model by:

            >>> self.prepare_data(...)
            >>> self.train_model(...)
        ----
        Example
        ----
        Genrally, during overriding this method, the process could be defined as follows:

            >>> out = self(x)
            >>> loss = criterion(out, y)
            >>> return loss
        *Warning:* The a scalar return format **loss** must be followed.

        ----
        How to applied Compiled method?
        ----
        If you want to use the complied model method provided by **PyTorch >= 2.0**. The outputs
        should be gotten by:
            >>> out = self.compiled(x)
        the compiled process will be applied automatically

        :return: a scaler value of loss.
        """
        model_out = self(x)
        loss = criterion(model_out.to(self.device), y)
        return loss

    def compiled(self, x: torch.Tensor) -> torch.Tensor:
        """
        Using the compiled module to computing. **PyTorch >= 2.0** is required.

        :param x: The input tensor.
        :return: Compiled output tensor
        """
        if self.opt_model is None:
            if int(torch.version.__version__.split('.')[0]) < 2:
                self.logger.error("The compiled method is only supported by PyTorch Version > 2.0,"
                                  "But found the version: {}".format(torch.version.__version__))
                return x
            else:
                self.logger.info("Compling model, it will take a few minutes...")
                self.opt_model = torch.compile(self, dynamic=False)
                torch.set_float32_matmul_precision('high')
                return self.opt_model(x)
        else:
            return self.opt_model(x)

    def epoch_start(self, epoch):
        """
        A callback function called before every training epoch starting.
        """
        self.train()

    def epoch_end(self, epoch):
        """
        A callback function called after every training epoch finished.
        """
        return

    def iter_start(self, iteration):
        """
        A callback function called before every iteration starting during one epoch.
        """
        return

    def iter_end(self, iteration):
        """
        A callback function called after every iteration end during one epoch.
        """
        return

    def iter_end_before_opt(self, iteration):
        """
        A callback function called after every iteration, before optimizer.step().
        """
        return

    def train_start(self):
        """
        A callback function called before training process starting.
        """
        return

    def train_end(self):
        """
        A callback function called after training process finished.
        """
        return

    def test_start(self):
        """
        A callback function called before testing process starting.
        """
        return

    def test_end(self):
        """
        A callback function called after testing process finished.
        """
        return
