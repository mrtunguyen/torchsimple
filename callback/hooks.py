from torch import nn, Tensor
from typing import Collection, Union, Callable, Any, Tuple, List
from torchsimple.utils import one_param, in_channels

Tensors = Union[Tensor, Collection['Tensors']]
HookFunc = Callable[[nn.Module, Tensors, Tensors], Any]
Sizes = List[List[int]]


def is_listy(x)->bool: return isinstance(x, (tuple, list))

class Hook():
    "Create a hook on `m` with `hook_func`."
    def __init__(self, m:nn.Module, hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hook_func,self.detach,self.stored = hook_func,detach,None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module:nn.Module, input:Tensors, output:Tensors):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input  = (o.detach() for o in input ) if is_listy(input ) else input.detach()
            output = (o.detach() for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed=True

    def __enter__(self, *args): return self
    def __exit__(self, *args): self.remove()
        
class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."
    def __init__(self, ms:Collection[nn.Module], hook_func:HookFunc, is_forward:bool=True, detach:bool=True):
        self.hooks = [Hook(m, hook_func, is_forward, detach) for m in ms]

    def __getitem__(self,i:int)->Hook: return self.hooks[i]
    def __len__(self)->int: return len(self.hooks)
    def __iter__(self): return iter(self.hooks)
    @property
    def stored(self): return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks: h.remove()

    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
        

def hook_output(module:nn.Module, detach:bool=True, grad:bool=False)->Hook:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, is_forward=not grad)

def hook_outputs(modules:Collection[nn.Module], detach:bool=True, grad:bool=False)->Hooks:
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)

def _hook_inner(m,i,o): return o if isinstance(o,Tensor) else o if is_listy(o) else list(o)

def dummy_batch(m: nn.Module, size:tuple=(64,64))->Tensor:
    "Create a dummy batch to go through `m` with `size`."
    ch_in = in_channels(m)
    return one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1.,1.)

def dummy_eval(m:nn.Module, size:tuple=(64,64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    return m.eval()(dummy_batch(m, size))

def model_sizes(m:nn.Module, size:tuple=(64,64))->Tuple[Sizes,Tensor,Hooks]:
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        x = dummy_eval(m, size)
        return [o.stored.shape for o in hooks]

def num_features_model(m:nn.Module)->int:
    "Return the number of output features for `model`."
    sz = 64
    while True:
        try: return model_sizes(m, size=(sz,sz))[-1][1]
        except Exception as e:
            sz *= 2
            if sz > 2048: raise
