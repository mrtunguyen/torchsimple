from functools import reduce
import sys
from typing import Any, Dict, Union, Hashable, Type, List, Optional

import numpy as np
from tqdm import tqdm
import h5py
import yaml
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from pathlib import Path
import os
from apex.fp16_utils import BN_convert_float

BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)

def load_h5(path:str):
    return h5py.File(path)

def load_yaml(path:str):
    with open(path, 'r') as file:
        res = yaml.load(file)
    return res

def freeeze_from_to(module: nn.Module,
                    n_from:int,
                    n_to: int,
                    freeze_bn: bool=False) -> None:
    layers = list(module.children())
    for l in layers[:n_from] or l in layers[n_to:]:
        for module in flatten_layer(l):
            if freeze_bn or not isinstance(module, BN_TYPES):
                set_grad(module, requires_grad = True)
    
    for l in layers[n_from:n_to]:
        for module in flatten_layer(l):
            if freeze_bn or not isinstance(module, BN_TYPES):
                set_grad(module, requires_grad = False)

def freeze_to(module: nn.Module,
              n: int,
              freeze_bn: bool = False) -> None:
    layers = list(module.children())
    for l in layers[:n]:
        for module in flatten_layer(l):
            if freeze_bn or not isinstance(module, BN_TYPES):
                set_grad(module, requires_grad=False)

    for l in layers[n:]:
        for module in flatten_layer(l):
            set_grad(module, requires_grad=True)


def freeze(module: nn.Module,
           freeze_bn: bool = False) -> None:
    freeze_to(module=module, n=-1, freeze_bn=freeze_bn)


def unfreeze(module: nn.Module) -> None:
    layers = list(module.children())
    for l in layers:
        for module in flatten_layer(l):
            set_grad(module, requires_grad=True)


def set_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad

def children(m:nn.Module):
    "Get children of 'm'"
    return list(m.children())

def num_children(m:nn.Module)->int:
    "Get number of children modules in 'm'"
    return len(children(m))
        
# https://github.com/fastai/fastai/
class ParameterModule(nn.Module):
    """Register a lone parameter `p` in a module."""
    def __init__(self, p: nn.Parameter):
        super().__init__()
        self.val = p

    def forward(self, x): return x


# https://github.com/fastai/fastai/
def children_and_parameters(m: nn.Module):
    """Return the children of `m` and its direct parameters not registered in modules."""
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],
                     [])
    for p in m.parameters():
        if id(p) not in children_p:
            children.append(ParameterModule(p))
    return children

flatten_model = lambda m: sum(map(flatten_model, children_and_parameters(m)), []) if num_children(m) else [m]

def in_channels(m:nn.Module)-> List[int]:
    for l in flatten_model(m):
        if hasattr(l, 'weight'): return l.weight.shape[1]
    raise Exception('No weight layer')

def flatten_layer(layer: nn.Module) -> List[nn.Module]:
    if len(list(layer.children())):
        layers = []
        for children in children_and_parameters(layer):
            layers += flatten_layer(children)
        return layers
    else:
        return [layer]

def one_param(m: nn.Module)-> Tensor:
    "Return the first parameter of 'm'"
    return next(m.parameters())

def to_numpy(data: torch.Tensor) -> np.ndarray:
    return data.detach().cpu().numpy()


def exp_weight_average(curr_val: Union[float, torch.Tensor],
                       prev_val: float,
                       alpha: float = 0.9) -> float:
    if isinstance(curr_val, torch.Tensor):
        curr_val = to_numpy(curr_val)
    return float(alpha * prev_val + (1 - alpha) * curr_val)


def get_pbar(dataloader: DataLoader,
             description: str) -> tqdm:

    pbar = tqdm(
        total=len(dataloader),
        leave=True,
        ncols=0,
        desc=description,
        file=sys.stdout)

    return pbar


def extend_postfix(postfix: str, dct: Dict) -> str:
    if postfix is None:
        postfix = ""
    postfixes = [postfix] + [f"{k}={v:.4f}" for k, v in dct.items()]
    return ", ".join(postfixes)


def get_opt_lr(opt: torch.optim.Optimizer) -> float:
    lrs = [pg["lr"] for pg in opt.param_groups]
    res = reduce(lambda x, y: x + y, lrs) / len(lrs)
    return res

def convert_model_to_half(model):
    old_model = model
    new_model = BN_convert_float(model.half())
    del old_model
    return new_model

class DotDict(dict):
    """
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr: str) -> Any:
        return self.get(attr)

    def __setattr__(self, key: Hashable, value: Any) -> Any:
        self.__setitem__(key, value)

    def __setitem__(self, key: Hashable, value: Any) -> Any:
        super().__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item: str) -> None:
        self.__delitem__(item)

    def __delitem__(self, key: str) -> None:
        super().__delitem__(key)
        del self.__dict__[key]


def load_state_dict(model: torch.nn.Module,
                    state_dict: Dict,
                    skip_wrong_shape: bool = False):
    model_state_dict = model.state_dict()

    for key in state_dict:
        if key in model_state_dict:
            if model_state_dict[key].shape == state_dict[key].shape:
                model_state_dict[key] = state_dict[key]
            elif not skip_wrong_shape:
                m = f"Shapes of the '{key}' parameters do not match: " \
                    f"{model_state_dict[key].shape} vs {state_dict[key].shape}"
                raise Exception(m)
            #TODO: if skip_wrong_shape
    model.load_state_dict(model_state_dict)
    
class SmoothValue():
    """Create a smooth moving average for a value (loss, etc) using 'beta'
    """ 
    def __init__(self, beta:float):
        self.beta, self.num, self.mov_arg = beta, 0, 0
        
    def add_value(self, value:float):
        self.num += 1
        self.mov_arg = self.beta * self.mov_arg + (1- self.beta) * value
        self.smooth = self.mov_arg / (1- self.beta ** self.num)