from torchsimple.lib import *
from .callback import Callback

__all__ = ['MixUpCallback', 'MixupLoss']

class MixUpCallback(Callback):
    def __init__(self, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):
        self.alpha, self.stack_x, self.stack_y = alpha, stack_x, stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.state.criterion = MixupLoss(self.state.criterion)
            
    def on_batch_begin(self, i:int, state:DotDict):
        if state.mode == "train":
            x = state.batch['inputs']
            y = state.batch['outputs']
            lambd = np.random.beta(self.alpha, self.alpha, y.size(0))
            lambd = np.concatenate([lambd[:, None], 1-lambd[:, None]], 1).max(1)
            shuffle = torch.randperm(y.size(0)).to(x.device)
            x1 , y1 = x[shuffle], y[shuffle]
            if self.stack_x:
                new_input = [x, x[shuffle], lambd]
            else:
                new_input = (x * lambd.view(lambd.size(0), 1, 1, 1) + x1 * (1 - lambd).view(lambd.size(0), 1, 1, 1))
            
            if self.stack_y:
                new_target = torch.cat([y[:, None].float(), y1[:, None].float(), lambd[:, None].float()], 1)
            else:
                if len(y.shape) == 2:
                    lambd = lambd.unsqueeze(1).float()
                new_target = y.float() * lambd + y1.float() * (1 - lambd)
            
            state.batch['inputs'] = new_input
            state.batch['outputs'] = new_target
            
    def on_train_end(self, **kwargs):
        self.state.criterion = self.state.criterion.get_old()
        
        
class MixupLoss(nn.Module):
    
    def __init__(self, crit, reduction = 'mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
            
        else:
            self.crit = partial(crit, reduction = 'none')
            self.old_crit = crit
            
        self.reduction = reduction
        
    def forward(self, output, target):
        if len(target.size()) == 2:
            loss1, loss2 = self.crit(output, target[:, 0].long()), self.crit(output, target[:, 1].long())
            d = (loss1 * target[:, 2] + loss2 * (1 - target[:, 2])).mean()
        else: d = self.crit(output, target)
        if self.reduction == 'mean': return d.mean()
        elif self.reduction == 'sum' : return d.sum()
        return d
    
    def get_old(self):
        if hasattr(self, 'old_crit'): return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit