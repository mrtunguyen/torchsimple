from typing import Any
from torchsimple.utils import DotDict
import torch

__all__ = ['Callback', 'Callbacks', 'LRUpdater']

    
class Callback:
    '''
    An abstract class that all callback(e.g., LossRecorder) classes
    extends from.
    Must be extended before usage.
    '''
    def on_train_begin(self, state: DotDict):
        "Initialize constants in the callback"
        pass

    def on_train_end(self, state: DotDict):
        "Cleaning up things and saving files/models"
        pass

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict):
        "At the beginning of each epoch"
        pass
  
    def on_epoch_end(self, epoch: int, state: DotDict):
        "Called at the end of an epoch"
        pass   
   
    def on_batch_begin(self, i: int, state: DotDict): 
        "Called at the beginning of the batch"
        pass
    
    def on_batch_end(self, i: int, state: DotDict): 
        "Called at the end of the batch"
        pass
    
    def on_backward_begin(self, state: DotDict):
        pass
    
    def on_backward_end(self, state: DotDict):
        """Called after backprop but before optimizer step.
        Useful for true weight decay in AdamW, clipnorm
        """
        pass

class Callbacks:
    def __init__(self, callbacks):
        if isinstance(callbacks, Callbacks):
            self.callbacks = callbacks.callbacks
        if isinstance(callbacks, list):
            self.callbacks = callbacks
        else:
            self.callbacks = []
    
    def on_batch_begin(self, i, state):
        for callback in self.callbacks:
            callback.on_batch_begin(i, state)
    
    def on_batch_end(self, i, state):
        for callback in self.callbacks:
            callback.on_batch_end(i, state)
    
    def on_epoch_begin(self, epoch, epochs, state):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, epochs, state)
        
    def on_epoch_end(self, epoch, state):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, state)
            
    def on_train_begin(self, state):
        for callback in self.callbacks:
            callback.on_train_begin(state)
            
    def on_train_end(self, state):
        for callback in self.callbacks:
            callback.on_train_end(state)
            
class LRUpdater(Callback):
    def __init__(self, beta:float=0.98) -> None:
        self.beta = beta
        
    def calc_lr(self) -> float:
        raise NotImplementedError
    
    def calc_momentum(self) -> float:
        raise NotImplementedError
        
    def update_lr(self, new_lr:float, optimizer: torch.optim.Optimizer) -> None:
        for pg in optimizer.param_groups:
            pg["lr"] = new_lr
        
    def update_momentum(self, new_momentum:float, optimizer: torch.optim.Optimizer) -> None:
        if "betas" in optimizer.param_groups[0]:
            for pg in optimizer.param_groups:
                pg["betas"] = (new_momentum, pg["betas"][1])
        else:
            for pg in optimizer.param_groups:
                pg["momentum"] = new_momentum
            
    def on_batch_end(self, i:int, state:DotDict) -> None:
        if state.mode == "train":
            new_lr = self.calc_lr()
            new_momentum = self.calc_momentum()
            if new_lr:
                self.update_lr(new_lr, state.opt)
            if new_momentum:
                self.update_momentum(new_momentum, state.opt)