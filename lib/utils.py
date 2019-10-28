from .common import *

def load_h5(path:str):
    return h5py.File(path)

def load_yaml(path:str):
    with open(path, 'r') as file:
        res = yaml.load(file)
    return res

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
        
class SmoothValue():
    """Create a smooth moving average for a value (loss, etc) using 'beta'
    """ 
    def __init__(self, beta:float):
        self.beta, self.num, self.mov_arg = beta, 0, 0
        
    def add_value(self, value:float):
        self.num += 1
        self.mov_arg = self.beta * self.mov_arg + (1- self.beta) * value
        self.smooth = self.mov_arg / (1- self.beta ** self.num)