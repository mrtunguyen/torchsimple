from torch import nn
import torch
from collections import OrderedDict
from typing import Union, Collection, List, Iterable

rnns = {
    'rnn' : nn.RNN,
    'lstm' : nn.LSTM,
    'gru' : nn.GRU
}
rnns_inv = dict((v, k) for k, v in rnns.items())

class BatchRNN(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 hidden_size:int, 
                 rnn_type:Union[str,Collection[str]]='gru', 
                 num_layers:int=1, 
                 bidirectional:bool=True, 
                 batch_norm:bool=True, 
                 sum_directions:bool=True, 
                 return_hidden:bool=False):
        
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = nn.BatchNorm1d(input_size) if batch_norm else None
        self.rnn = rnns[rnn_type.lower()](input_size = input_size, hidden_size = hidden_size,
                                         bidirectional = bidirectional, bias = True)
        self.sum_directions = sum_directions
        self.return_hidden = return_hidden
        
    def flatten_parameters(self):
        self.rnn.flatten_parameters()
        
    def forward(self, x):
        if self.batch_norm is not None:
            x = x._replace(data = self.batch_norm(x.data))
        x, h = self.rnn(x)
        if self.sum_directions and self.bidirectional:
            x = x._replace(data=x.data[:, :self.hidden_size] + x.data[:, self.hidden_size:]) #sum bidirectional outputs
            
        if self.return_hidden: return x, h
        else: return x
    
class DeepBatchRNN(nn.Module):
    def __init__(self, 
                 input_size:int, 
                 hidden_size:int,
                 num_rnn:int=2,
                 rnn_type:Union[str,Collection[str]]='gru',
                 num_layers:Union[int, Collection[int]]=1,
                 bidirectional:Union[bool, Collection[bool]]=[True, True],
                 batch_norm:Union[bool, Collection[bool]]=[True, True],
                 sum_directions:Union[bool, Collection[bool]]=[True, False]):
        
        super(DeepBatchRNN, self).__init__()    
        num_layers = check_len_type(num_layers, num_rnn, int)
        rnn_type = check_len_type(rnn_type, num_rnn, str)
        hidden_size = check_len_type(hidden_size, num_rnn, int)
        bidirectional = check_len_type(bidirectional, num_rnn, bool)
        batch_norm = check_len_type(batch_norm, num_rnn, bool)
        sum_directions = check_len_type(sum_directions, num_rnn, bool)
            
        rnn_input_size = [input_size]
        for hidden_layer_size, sum_direction in zip(hidden_size[:-1], sum_directions[:-1]):
            if sum_direction: rnn_input_size += [hidden_layer_size]
            else: rnn_input_size += [2*hidden_layer_size]
            
        rnns = []
        for i in range(num_rnn):
            rnn = BatchRNN(input_size=rnn_input_size[i], 
                           hidden_size=hidden_size[i],
                           rnn_type = rnn_type[i],
                           num_layers = num_layers[i],
                           bidirectional = bidirectional[i],
                           batch_norm=batch_norm[i],
                           sum_directions = sum_directions[i],
                           return_hidden = False)
            rnns.append((f'{i}', rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
    
    def flatten_parameters(self):
        for rnn in self.rnns:
            rnn.flatten_parameters()
        
    def forward(self, x):
        x = self.rnns(x)
        return x
            
def check_len_type(a, length, type_a):
    if isinstance(a, type_a): a = [a] * length
    elif isinstance(a, List): assert len(a) == length, f'mismatch length. Expected {length} found {len(a)}'
    else: raise Exception(f'{a} must be type {type_a} or List of {type_a}')
    return a