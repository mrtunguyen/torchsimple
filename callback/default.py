from apex import amp
from apex.amp._amp_state import _amp_state
from pathlib import Path
from collections import defaultdict
import shutil
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from typing import Optional, Callable, Union, Dict, List, Type
from .callback import Callback
from torchsimple.utils import DotDict, get_pbar, to_numpy, exp_weight_average, extend_postfix

__all__ = ['DefaultLossCallback', 'DefaultOptimizerCallback', 'DefaultSchedulerCallback',
           'DefaultMetricsCallback', 'PredictionsSaverCallback', 'CheckpointSaverCallback',
           'ProgressBarCallback', 'EarlyStoppingCallback']

class DefaultLossCallback(Callback):
    """
    A callback to calculate loss by default.
    If a loss is not precised during fit of training, this callback will be 
    used for the loss of model.
    """
    def __init__(self, target_key: str, preds_key: str) -> None:
        self.target_key = target_key
        self.preds_key = preds_key

    def on_batch_end(self, i: int, state: DotDict) -> None:
        target = state.batch[self.target_key]
        preds = state.out[self.preds_key]
        state.loss = state.criterion(preds, target)
    

class DefaultOptimizerCallback(Callback):
    """
    A callback to set Optimizer by default.
    If an optimizer is not precised during fit of training, this callback will be 
    used for the optimizer of model.
    """
    def __init__(self, 
                 clip:float=1.0,
                 loss_key:str='loss',
                 accum_steps:int=1):
        self.clip = clip
        self.loss_key = loss_key
        self.accum_steps = accum_steps
        
    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.mode == "train":
            
            state.opt.zero_grad()
            if isinstance(state.loss, torch.Tensor):
                loss = state.loss
            elif isinstance(state.loss, dict):
                loss = state.loss[self.loss_key]
            
            if self.accum_steps:
                loss = loss / self.accum_steps
                
            if state.use_fp16:
                state.opt.backward(loss)
            else:
                loss.backward()
                
            self.on_backward_end(state)
            
            _amp_state.verbosity = 0 #to mute the message loss scale of apex
            if (i+1) % self.accum_steps == 0:
                state.opt.step()
                state.opt.zero_grad()
                
    def on_backward_end(self, state: DotDict):
        "Clip the gradient before the optimizer step"
        if self.clip: 
            if state.use_fp16:
                state.opt.clip_master_grads(self.clip)
            else:
                torch.nn.utils.clip_grad_norm_(state.model.parameters(), self.clip)
            
class DefaultSchedulerCallback(Callback):
    """
    A callback to set scheduler of learning rate by default.
    If a scheduler is not precised during fit of training, this callback will be 
    used for the scheduler of model.
    """
    def __init__(self,
                 sched: Union[_LRScheduler, ReduceLROnPlateau],
                 metric = "val_loss") -> None:
        self.metric = metric
        if isinstance(sched, ReduceLROnPlateau):
            self.when = "on_epoch_end"
        else:
            self.when = "on_epoch_begin"

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        if self.when == "on_epoch_begin" and state.mode == "train":
            state.sched.step()

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if self.when == "on_epoch_end" and state.mode == "train":
            state.sched.step(state.epoch_metrics[self.metric])


class DefaultMetricsCallback(Callback):
    """
    A callback to set metric by default.
    If a metric is not precised during fit of training, this callback will be 
    used for the metric of model.
    """
    def __init__(self,
                 target_key: str,
                 preds_key: str,
                 metrics: Optional[Dict[str, Callable]] = None,
                 loss_key : str = 'loss') -> None:
        self.metrics = metrics or {}
        self.pbar_metrics = None
        self.target_key = target_key
        self.preds_key = preds_key
        self.loss_key = loss_key
    
    def get_metric(self,
                   metric : Callable,
                   target : torch.Tensor, 
                   preds : Union[List, torch.Tensor]):
        if isinstance(preds, list):
            preds = torch.cat(preds)
        return metric(target, preds)
    
    def update_epoch_metrics(self,
                             target: Type[torch.Tensor],
                             preds: Type[torch.Tensor]) -> None:
        for name, m in self.metrics.items():
            value = self.get_metric(m, target, preds)
            self.pbar_metrics[name] += value

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        self.pbar_metrics = defaultdict(float)

    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.mode == "val":
            if isinstance(state.loss, dict):
                loss = state.loss[self.loss_key]
            elif isinstance(state.loss, torch.Tensor):
                loss = state.loss
                
            self.pbar_metrics["val_loss"] += float(to_numpy(loss))
            self.update_epoch_metrics(target=state.batch[self.target_key],
                                      preds=state.out[self.preds_key])
        # tb logs
        if state.mode != "test" and state.do_log:
            if isinstance(state.loss, dict):
                for key, value in state.loss.items():
                    state.metrics[state.mode][key] = float(to_numpy(state.loss[key]))
            elif isinstance(state.loss, torch.Tensor):
                state.metrics[state.mode]["loss"] = float(to_numpy(state.loss))
            
            for name, m in self.metrics.items():
                preds = state.out[self.preds_key]
                target = state.batch[self.target_key]
                value = self.get_metric(m, target, preds)
                state.metrics[state.mode][name] = value

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        divider = len(state.loader)
        for k in self.pbar_metrics.keys():
            self.pbar_metrics[k] /= divider
        state.epoch_metrics = self.pbar_metrics
        

class PredictionsSaverCallback(Callback):
    def __init__(self,
                 savepath: Optional[Union[str, Path]],
                 preds_key: str) -> None:
        if savepath is not None:
            self.savepath = Path(savepath)
            self.savepath.parent.mkdir(exist_ok=True)
            self.return_array = False
        else:
            self.savepath = None
            self.return_array = True
        
        self.preds_key = preds_key
        self.preds = []

    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.mode == "test":
            out = state.out[self.preds_key]
            # DataParallelModel workaround
            if isinstance(out, list):
                out = np.concatenate([to_numpy(o) for o in out])
            else:
                out = to_numpy(out)
            self.preds.append(out)

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.mode == "test":
            preds = np.concatenate(self.preds)
            if self.return_array:
                state.preds = self.preds_key
            else:
                np.save(self.savepath, preds)
            self.preds = []


class CheckpointSaverCallback(Callback):
    def __init__(self,
                 savedir: str,
                 iterators_per_checkpoint: int = None,
                 metric: Optional[str] = None,
                 n_best: int = 3,
                 prefix: Optional[str] = None,
                 mode: str = "min") -> None:
        self.metric = metric or "val_loss"
        self.n_best = n_best
        self.savedir = Path(savedir)
        self.savedir.parent.mkdir(exist_ok=True)
        self.iterators_per_checkpoint = iterators_per_checkpoint
        self.prefix = f"{prefix}." if prefix is not None else "checkpoint."

        if mode not in ["min", "max"]:
            raise ValueError(f"mode should be 'min' or 'max', got {mode}")
        if mode == "min":
            self.maximize = False
        if mode == "max":
            self.maximize = True

        self.best_scores = []
        
    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        # trim best scores
        self.best_scores = self.best_scores[:epochs]

    def on_batch_end(self, i:int, state:DotDict)-> None:
        if self.iterators_per_checkpoint and i == self.iterators_per_checkpoint:
            checkpoint_name = f"{self.prefix}iter{i}.h5"
            state.checkpoint = f"{self.savedir / checkpoint_name}"
    
    
    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.mode == "val":
            score_val = state.epoch_metrics[self.metric]
            score_name = f"{self.prefix}epoch_{epoch + 1}.h5"
            score = (score_val, score_name)
            sorted_scores = sorted(self.best_scores + [score],
                                   reverse=self.maximize)
            self.best_scores = sorted_scores[:self.n_best]
            # set_trace()
            if score_name in (s[1] for s in self.best_scores):
                state.checkpoint = f"{self.savedir / score_name}"
                #remove the last saved checkpoint if get better score
                if len(sorted_scores) > self.n_best:
                    # set_trace()
                    Path(f"{self.savedir / sorted_scores[-1][1]}").unlink()

    def on_train_end(self, state: DotDict) -> None:
        if state.mode == "val":
            best_cp = self.savedir / self.best_scores[0][1]
            shutil.copy(str(best_cp), f"{self.savedir}/{self.prefix}best.h5")
            print(f"\nCheckpoint\t{self.metric or 'val_loss'}")
            for score in self.best_scores:
                print(f"{self.savedir/score[1]}\t{score[0]:.6f}")


class EarlyStoppingCallback(Callback):
    def __init__(self,
                 patience: int,
                 metric: Optional[str] = None,
                 mode: str = "min",
                 min_delta: int = 0) -> None:
        self.best_score = None
        self.metric = metric or "val_loss"
        self.patience = patience
        self.num_bad_epochs = 0
        self.is_better = None

        if mode not in ["min", "max"]:
            raise ValueError(f"mode should be 'min' or 'max', got {mode}")
        if mode == "min":
            self.is_better = lambda score, best: score <= (best - min_delta)
        if mode == "max":
            self.is_better = lambda score, best: score >= (best - min_delta)

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.mode == "val":
            score = state.epoch_metrics[self.metric]
            if self.best_score is None:
                self.best_score = score
            if self.is_better(score, self.best_score):
                self.num_bad_epochs = 0
                self.best_score = score
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                state.stop_train = True
                  
class ProgressBarCallback(Callback):
    def __init__(self) -> None:
        self.running_loss = None
        self.val_loss = None

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        self.running_loss = None
        self.val_loss = None

        loader = state.loader
        if state.mode == "train":
            description = f"TRAIN | Epoch {epoch+1}/{epochs}"
            state.pbar = get_pbar(loader, description)
        elif state.mode == "val":
            description = f"VALID | Epoch {epoch + 1}/{epochs}"
            state.pbar = get_pbar(loader, description)
        elif state.mode == "test":
            description = "Predict"
            state.pbar = get_pbar(loader, description)

    def on_batch_end(self, i: int, state: DotDict) -> None:

        if state.mode == "train":
            if not self.running_loss:
                self.running_loss = to_numpy(state.loss)

            self.running_loss = exp_weight_average(state.loss,
                                                   self.running_loss)
            postfix = {"loss": f"{self.running_loss:.4f}"}
            state.pbar.set_postfix(postfix)
            state.pbar.update()
            
        elif state.mode == "val":
            if not self.val_loss:
                self.val_loss = to_numpy(state.loss)
            self.val_loss = exp_weight_average(state.loss,
                                               self.val_loss)
            postfix = {"loss": f"{self.val_loss:.4f}"}
            state.pbar.set_postfix(postfix)
            state.pbar.update()
            
        elif state.mode == "test":
            state.pbar.update()

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.mode == "train":
            losses = state.get("epoch_metrics", {})
            state.pbar.set_postfix_str(extend_postfix(state.pbar.postfix,
                                                          losses))
            state.pbar.close()
        
        if state.mode == "val":
            metrics = state.get("epoch_metrics", {})
            state.pbar.set_postfix_str(extend_postfix(state.pbar.postfix,
                                                      metrics))
            state.pbar.close()
        elif state.mode == "test":
            state.pbar.close()