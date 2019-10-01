# from tensorboardX import SummaryWriter
from pathlib import Path
import numpy as np
from typing import Union
from collections import defaultdict
from .callback import Callback
from torchsimple.core import MonitorWriter
from torchsimple.utils import DotDict, get_opt_lr, to_numpy
import shutil
__all__ = ['Logger']

class Logger(Callback):
    def __init__(self, logdir: Union[str, Path]) -> None:
        self.logdir = Path(logdir)
        self.writer = None
        self.total_iter = 0
        self.train_iter = 0
        self.val_iter = 0
        self.val_batch_iter = 0
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
    def update_total_iter(self, mode: str) -> None:
        if mode == "train":
            self.train_iter += 1
            self.train_batch_iter += 1
        if mode == "val":
            self.val_iter += 1
            self.val_batch_iter +=1
        self.total_iter += 1

    def on_train_begin(self, state: DotDict) -> None:
        self.train_iter = 0
        self.val_iter = 0
        if self.logdir.exists(): shutil.rmtree(self.logdir)
        self.logdir.mkdir(exist_ok=True)
        self.train_epoch = defaultdict(list)
        self.val_epoch = defaultdict(list)
        try:
            from tensorboardX import SummaryWriter
            self.is_tensorboard_available = True
            self.train_writer = SummaryWriter(str(self.logdir / "train"))
            self.val_writer = SummaryWriter(str(self.logdir / "val"))
            self.metric_writer = SummaryWriter(str(self.logdir / "metric"))
        except ImportError:
            print("Dont have tensorboard. Will use log to monitor training")
            self.is_tensorboard_available = False
            self.train_writer = MonitorWriter(logdir = self.logdir, filename = 'train_loss.txt')
            self.train_writer.reset()
            self.train_writer.update('loss,lr', 0)
            self.val_writer = MonitorWriter(logdir = self.logdir, filename = 'val_loss.txt')
            self.val_writer.reset()
            self.val_writer.update('loss', 0)
            self.metric_writer = MonitorWriter(self.logdir, 'metrics.txt')
            self.metric_writer.reset()

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict):
        self.train_batch_iter = 0
        self.val_batch_iter = 0

    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.core.mode == "train":
            
            lr = get_opt_lr(state.core.opt)
            if self.is_tensorboard_available:
                for name, metric in state.core.metrics["train"].items():
                    self.train_writer.add_scalar(f"batch/{name}",
                                                 float(metric),
                                                 global_step=self.total_iter)
                    self.train_epoch[name].append(float(metric))
                    
                self.train_writer.add_scalar("batch/lr",
                                             float(lr),
                                             global_step=self.train_iter)
                self.train_writer.add_scalar("batch/loss_train",
                                             float(to_numpy(state.core.loss)),
                                             global_step=self.train_iter)
                self.train_epoch['loss'].append(float(to_numpy(state.core.loss)))
                
            else:
                self.train_writer.update(','.join([str(to_numpy(state.core.loss)), str(lr)]), i+1)
            self.update_total_iter(state.core.mode)
        elif state.core.mode == "val":
            if self.is_tensorboard_available:
                for name, metric in state.core.metrics["val"].items():
                    self.val_writer.add_scalar(f"batch/{name}",
                                               float(metric),
                                               global_step=self.total_iter)
                    self.val_epoch[name].append(float(metric))
                self.val_epoch['loss'].append(float(to_numpy(state.core.loss)))
            
            else:
                self.val_writer.update(str(to_numpy(state.core.loss)), i+1)
                self.metric_writer.update(','.join([str(x) for x in list(state.core.metrics['val'].values())]), i+1)
                
            self.update_total_iter(state.core.mode)
            
    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.core.mode == "train":
            if self.is_tensorboard_available:
                for name, value in self.train_epoch.items():
                    mean = np.mean(value[-self.train_batch_iter:]) 
                    self.train_writer.add_scalar(f"epoch/{name}",
                                                 float(mean),
                                                 global_step=epoch)
        if state.core.mode == "val":
            if self.is_tensorboard_available:
                for name, value in self.val_epoch.items():
                    mean = np.mean(value[-self.val_batch_iter:])  # last epochs vals
                    self.val_writer.add_scalar(f"epoch/{name}",
                                           float(mean),
                                           global_step=epoch)
                
    def on_train_end(self, state: DotDict) -> None:
        self.train_writer.close()
        self.val_writer.close()