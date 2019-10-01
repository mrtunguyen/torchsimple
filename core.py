import os
from typing import List, Iterable
import numpy as np
from functools import partial
from torch import optim

AdamW = partial(optim.Adam, betas=(0.9,0.99))

def is_tuple(x)->bool: return isinstance(x, tuple)

def listify(p=None, q=None):
    if p is None: p = []
    elif isinstance(p, str): p = [p]
    elif not isinstance(p, Iterable): p = [p]
    else: 
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p*n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)

def generate_maps(labels:List):
    token_to_index = dict([(token, index) for index, token in enumerate(labels)])
    index_to_token = dict([(index, token) for index, token in enumerate(labels)])
    return token_to_index, index_to_token

def annealing_exp(start:int, end:int, percentage:float):
    return start * (end/start) ** percentage

def annealing_cos(start:int, end:int, percentage:float):
    cos_out = np.cos(np.pi * percentage) + 1
    return end + (start - end)/2 * cos_out

def annealing_linear(start:float, end:float, percentage:float):
    "Linearly anneal from start to end as percentage goes from 0.0 to 1.0"
    return start + percentage * (end-start)
    
def no_annealing(start:float):
    "No annealing"
    return start

class Scheduler():
    
    def __init__(self, values, n_iter:int, func=None):
        self.start, self.end = (values[0], values[1]) if is_tuple(values) else (values, 0)
        self.n_iter = max(1, n_iter)
        if func is None: 
            self.func = annealing_linear if is_tuple(values) else no_annealing
        else:
            self.func = func
            
        self.n = 0
        
    def restart(self): self.n = 0
        
    def step(self):
        "Return next value along annealed schedule"
        self.n += 1
        return self.func(self.start, self.end, self.n/self.n_iter)
    
    @property
    def is_done(self):
        return self.n >= self.n_iter
    
class AverageMeter(object):
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = 0
        self.count = 0
    
    def update(self, val):
        self.count += 1
        self.sum += val 
        self.val[self.count] = val
        self.avg[self.count] = self.sum/self.count
        
class MonitorWriter(object):
    def __init__(self, logdir, filename):
        self.logdir = logdir
        self.filename = filename
        self.file = open(os.path.join(logdir, filename), mode = 'a')
        
    def reset(self):
        self.file = open(os.path.join(self.logdir, self.filename), mode = 'w')
        
    def update(self, loss, iter):
        self.file.write(f"{iter},{loss}\n")
        
    def close(self):
        self.file.close()