import numpy as np
import torch 
import shutil
import gc
import matplotlib.pyplot as plt
from pathlib import Path
from torchsimple.utils import load_state_dict, SmoothValue, DotDict, to_numpy
from torchsimple.core import Scheduler, annealing_exp, annealing_cos
from torchsimple.parallel import DataParallelModel
from .callback import LRUpdater

__all__ = ['LRFinder']

class LRFinder(LRUpdater):
    """
    Find an optimal learning rate for a model at the beginning of training,
    Learning rate is increased in log scale
    """
    def __init__(self, final_lr:float=5., 
                       n_steps:int=200, 
                       init_lr:float=1e-6, 
                       stop_div:bool=True):
        super().__init__()
        self.init_lr = init_lr
        self.smoother = SmoothValue(self.beta)
        self.sched = Scheduler((init_lr, final_lr), n_steps, annealing_exp)
        
    def calc_lr(self) -> float:
        self.new_lr = self.sched.step()
        return self.new_lr
    
    def calc_momentum(self) -> float:
        pass
    
    def update_momentum(self, optimizer: torch.optim.Optimizer) -> None:
        pass
    
    def on_train_begin(self, state: DotDict) -> None:
        self.loss_dict = {'iteration' : [], 'loss' : [], 'smooth_loss' : [], 'lr' : []}
        self.best_loss = 0
        Path('tmp').mkdir(exist_ok = True)
        torch.save(state.model.state_dict(), 'tmp/tmp.pth')
        self.update_lr(self.init_lr, state.opt)
        
    def on_batch_end(self, i: int, state: DotDict) -> None:
        super().on_batch_end(i, state)
        self.smoother.add_value(to_numpy(state.loss))
        smooth_loss = self.smoother.smooth
        if i == 0 or smooth_loss < self.best_loss : self.best_loss = smooth_loss
        
        self.loss_dict['smooth_loss'].append(smooth_loss)
        self.loss_dict['loss'].append(to_numpy(state.loss))
        self.loss_dict['lr'].append(self.new_lr)
        if self.sched.is_done or (self.smoother.smooth > 4 * self.best_loss or np.isnan(smooth_loss)):
            state.stop_epoch = True
    
    def plot(self):
        lrs = self.loss_dict['lr']
        loss = self.loss_dict['smooth_loss']
        plt.figure(figsize= (20, 5))
        plt.subplot(1, 2, 1)
        plt.plot(lrs)
        plt.xlabel('iterations')
        plt.ylabel('learning rate')
        
        plt.subplot(1, 2, 2)
        plt.plot(lrs, loss)
        plt.xlabel('learning rate')
        plt.ylabel('loss')
        plt.show()
    
    def on_train_end(self, state: DotDict) -> None:
        "Cleanup learn model weights disturbed during LRFinder"
        checkpoint = torch.load('tmp/tmp.pth', map_location=lambda storage, loc: storage)
        if not isinstance(state.model, DataParallelModel) \
                and "module." in list(checkpoint.keys())[0]:
            # [7:] is to skip 'module.' in group name
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}

        load_state_dict(model=state.model,
                        state_dict=checkpoint,
                        skip_wrong_shape=False)
        if hasattr(state.model, 'reset'): state.model.reset()
        shutil.rmtree('tmp/')
        del checkpoint
        gc.collect()
        self.plot()