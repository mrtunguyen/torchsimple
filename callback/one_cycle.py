from .callback import LRUpdater
from torchsimple.lib import *

__all__ = ['OneCycleLR']

class OneCycleLR(LRUpdater):
    def __init__(self, 
                 max_lr: float,
                 momentum_range: Tuple[float, float]=(0.95,0.85),
                 div_factor: float=25.,
                 increase_fraction: float=0.3,
                 final_div:float=None,
                 total_epochs:int=None,
                 start_epoch:int=None) -> None:
        super().__init__()
        self.max_lr = max_lr
        self.momentum_range = momentum_range
        self.div_factor = div_factor
        self.increase_fraction = increase_fraction
        if final_div is None: 
            self.final_div = div_factor*1e4
        else:
            self.final_div = final_div
        # point in iterations for starting lr decreasing
        self.total_epochs, self.start_epoch = total_epochs, start_epoch
    
    def steps(self, *steps_cfg):
        return [Scheduler(step, n_iter, func=func) for (step, (n_iter, func)) in zip(steps_cfg, self.phases)]
    
    def on_train_begin(self, state: DotDict):
        res = {'epoch' : self.start_epoch} if self.start_epoch is not None else None
        n = len(state.dataowner.train_dl) * self.total_epochs
        a1 = int(n * self.increase_fraction)
        a2 = n - a1
        self.phases = ((a1, annealing_cos), (a2, annealing_cos))
        low_lr = self.max_lr/self.div_factor
        self.lr_scheds = self.steps((low_lr, self.max_lr), (self.max_lr, self.max_lr/self.final_div))
        self.mom_scheds = self.steps(self.momentum_range, (self.momentum_range[1], self.momentum_range[0]))
        self.idx_s = 0
        self.update_lr(self.lr_scheds[0].start, state.opt)
        self.update_momentum(self.mom_scheds[0].start, state.opt)
        
    def calc_lr(self) -> float:
        return self.lr_scheds[self.idx_s].step()
    
    def calc_momentum(self) -> float:
        return self.mom_scheds[self.idx_s].step()
    
    def on_batch_end(self, i:int, state:DotDict):
        super().on_batch_end(i, state)
        if state.mode == "train" and self.lr_scheds[self.idx_s].is_done:
            self.idx_s += 1