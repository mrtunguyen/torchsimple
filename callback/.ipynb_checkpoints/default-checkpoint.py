from apex import amp
from pathlib import Path
import shutil
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
        target = state.core.batch[self.target_key]
        preds = state.core.out[self.preds_key]
        state.loss = state.core.criterion(preds, target)
    

class DefaultOptimizerCallback(Callback):
    """
    A callback to set Optimizer by default.
    If an optimizer is not precised during fit of training, this callback will be 
    used for the optimizer of model.
    """
    def __init__(self, clip:float=None):
        self.clip = clip
        
    def on_batch_end(self, i: int, state: DotDict) -> None:
        if state.core.mode == "train":
            state.core.opt.zero_grad()
            if state.core.use_fp16:
                state.core.opt.backward(state.core.loss)
            else:
                state.core.loss.backward()
                
            self.on_backward_end(state)
            state.core.opt.step()
    
    def on_backward_end(self, state: DotDict):
        "Clip the gradient before the optimizer step"
        if self.clip: 
            if state.core.use_fp16:
                state.core.opt.clip_master_grads(self.clip)
            else:
                torch.nn.utils.clip_grad_norm_(state.core.model.parameters(), self.clip)
            
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
        if self.when == "on_epoch_begin" and state.core.mode == "train":
            state.core.sched.step()

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if self.when == "on_epoch_end" and state.core.mode == "train":
            state.core.sched.step(state.core.epoch_metrics[self.metric])


class DefaultMetricsCallback(Callback):
    """
    A callback to set metric by default.
    If a metric is not precised during fit of training, this callback will be 
    used for the metric of model.
    """
    def __init__(self,
                 target_key: str,
                 preds_key: str,
                 metrics: Optional[Dict[str, Callable]] = None) -> None:
        self.metrics = metrics or {}
        self.pbar_metrics = None
        self.target_key = target_key
        self.preds_key = preds_key
    
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
        if state.core.mode == "val":
            self.pbar_metrics["val_loss"] += float(to_numpy(state.core.loss))
            self.update_epoch_metrics(target=state.core.batch[self.target_key],
                                      preds=state.core.out[self.preds_key])
        # tb logs
        if state.core.mode != "test" and state.core.do_log:
            state.core.metrics[state.core.mode]["loss"] = float(to_numpy(state.core.loss))
            for name, m in self.metrics.items():
                preds = state.core.out[self.preds_key]
                target = state.core.batch[self.target_key]
                value = self.get_metric(m, target, preds)
                state.core.metrics[state.core.mode][name] = value

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        divider = len(state.core.loader)
        for k in self.pbar_metrics.keys():
            self.pbar_metrics[k] /= divider
        state.core.epoch_metrics = self.pbar_metrics
        

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
        if state.core.mode == "test":
            out = state.core.out[self.preds_key]
            # DataParallelModel workaround
            if isinstance(out, list):
                out = np.concatenate([to_numpy(o) for o in out])
            else:
                out = to_numpy(out)
            self.preds.append(out)

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.core.mode == "test":
            preds = np.concatenate(self.preds)
            if self.return_array:
                state.core.preds = preds_key
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
            state.core.checkpoint = f"{self.savedir / checkpoint_name}"
    
    
    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.core.mode == "val":
            score_val = state.core.epoch_metrics[self.metric]
            score_name = f"{self.prefix}epoch_{epoch + 1}.h5"
            score = (score_val, score_name)
            sorted_scores = sorted(self.best_scores + [score],
                                   reverse=self.maximize)
            self.best_scores = sorted_scores[:self.n_best]
            # set_trace()
            if score_name in (s[1] for s in self.best_scores):
                state.core.checkpoint = f"{self.savedir / score_name}"
                #remove the last saved checkpoint if get better score
                if len(sorted_scores) > self.n_best:
                    # set_trace()
                    Path(f"{self.savedir / sorted_scores[-1][1]}").unlink()

    def on_train_end(self, state: DotDict) -> None:
        if state.core.mode == "val":
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
        if state.core.mode == "val":
            score = state.core.epoch_metrics[self.metric]
            if self.best_score is None:
                self.best_score = score
            if self.is_better(score, self.best_score):
                self.num_bad_epochs = 0
                self.best_score = score
            else:
                self.num_bad_epochs += 1

            if self.num_bad_epochs >= self.patience:
                state.core.stop_train = True
                  
class ProgressBarCallback(Callback):
    def __init__(self) -> None:
        self.running_loss = None
        self.val_loss = None

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        self.running_loss = None
        self.val_loss = None

        loader = state.core.loader
        if state.core.mode == "train":
            description = f"TRAIN | Epoch {epoch+1}/{epochs}"
            state.core.pbar = get_pbar(loader, description)
        elif state.core.mode == "val":
            description = f"VALID | Epoch {epoch + 1}/{epochs}"
            state.core.pbar = get_pbar(loader, description)
        elif state.core.mode == "test":
            description = "Predict"
            state.core.pbar = get_pbar(loader, description)

    def on_batch_end(self, i: int, state: DotDict) -> None:

        if state.core.mode == "train":
            if not self.running_loss:
                self.running_loss = to_numpy(state.core.loss)

            self.running_loss = exp_weight_average(state.core.loss,
                                                   self.running_loss)
            postfix = {"loss": f"{self.running_loss:.4f}"}
            state.core.pbar.set_postfix(postfix)
            state.core.pbar.update()
            
        elif state.core.mode == "val":
            if not self.val_loss:
                self.val_loss = to_numpy(state.core.loss)
            self.val_loss = exp_weight_average(state.core.loss,
                                               self.val_loss)
            postfix = {"loss": f"{self.val_loss:.4f}"}
            state.core.pbar.set_postfix(postfix)
            state.core.pbar.update()
            
        elif state.core.mode == "test":
            state.core.pbar.update()

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if state.core.mode == "train":
            losses = state.core.get("epoch_metrics", {})
            state.core.pbar.set_postfix_str(extend_postfix(state.core.pbar.postfix,
                                                          losses))
            state.core.pbar.close()
        
        if state.core.mode == "val":
            metrics = state.core.get("epoch_metrics", {})
            state.core.pbar.set_postfix_str(extend_postfix(state.core.pbar.postfix,
                                                      metrics))
            state.core.pbar.close()
        elif state.core.mode == "test":
            state.core.pbar.close()