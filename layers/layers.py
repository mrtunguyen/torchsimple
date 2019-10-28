from lib import *

NormType = Enum('NormType', 'Batch BatchZero Weight Spectral')

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def forward(self, x): 
        return self.func(x)
    
class View(nn.Module):
    def __init__(self, *size):
        super().__init__()
        self.size = size
        
    def forward(self, x):
        return x.view(self.size)
    
class Flatten(nn.Module):
    "Flatten `x` to a single dimension, often used at the end of a model. `full` for rank-1 tensor"
    def __init__(self, full:bool=False):
        super().__init__()
        self.full = full
    
    def forward(self, x):
        return x.view(-1) if self.full else x.view(x.size(0), -1)

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, size:Optional[int]=None):
        super().__init__()
        self.output_size = size or 1
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)
    
    def forward(self, x): 
        return torch.cat([self.mp(x), self.ap(x)], axis = 1)

    
def init_default(m:nn.Module, func=nn.init.kaiming_normal_):
    "Initialize 'm' weights with 'func' and set 'bias' to 0"
    if func:
        if hasattr(m, 'weight'): func(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
    return m

def bn_dropout_linear(n_in:int, n_out:int, bn:bool=True, p_dropout:float=0, actn:Optional[nn.Module]=None):
    "Sequence of batchnorm , dropout, and linear layers followed by actn"
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p_dropout != 0: layers.append(nn.Dropout(p_dropout))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

def create_head(n_features:int, n_class:int, linear_features:Optional[Collection[int]]=None,
                p_dropout:float=0.5, concat_pool:bool=True, bn_final:bool=False):
    "Model head that takes n_features features, run through 'Pool -> Flatten -> (bn -> FC -> act)' and about n_class classes"
    linear_features = [n_features, 512, n_class] if linear_features is None else [n_features] + linear_features + [n_class]
    p_dropout = listify(p_dropout)
    if len(p_dropout) == 1: p_dropout = [p_dropout[0]/2] * (len(linear_features) - 2) + p_dropout
    activations = [nn.ReLU(inplace=True)] * (len(linear_features) -2) + [None]
    pool = AdaptiveConcatPool2d(n_features) if concat_pool else nn.AdaptiveAvgPool2d(n_features)
    layers = [pool, Flatten()]
    for n_in, n_out, p, activation in zip(linear_features[:-1], linear_features[1:], p_dropout, activations):
        layers += bn_dropout_linear(n_in, n_out, True, p, activation)
    if bn_final: layers.append(nn.BatchNorm1d(linear_features[-1], momentum=0.01))
    return nn.Sequential(*layers)
    

def relu(inplace:bool=False, leaky:float=None):
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)


def conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None, type_conv:str='conv2d',
               norm_type=NormType.Batch, use_activ:bool=True, leaky:float=None, transpose:bool=False,
               init=nn.init.kaiming_normal_, self_attention:bool=False):
    conv_dict = {'conv1d' : nn.Conv1d,
                'conv2d' : nn.Conv2d,
                'convtranspose2d' : nn.ConvTranspose2d}
    batchnorm_dict = {'conv1d' : nn.BatchNorm1d,
                      'conv2d' : nn.BatchNorm2d}
    
    if padding is None: padding = (ks-1)//2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None: bias = not bn
    conv_func = conv_dict[type_conv]
    conv = init_default(conv_func(ni, nf, kernel_size = ks, bias = bias, stride = stride, padding = padding), init)
    if norm_type == NormType.Weight: conv = weight_norm(conv)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append(batchnorm_dict[type_conv](nf))
    if self_attention: layers.append(SelfAttention(nf))
    
    return nn.Sequential(*layers)

class SequentialEx(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        res = x
        for layer in self.layers:
            res.orig = x
            nres = layer(res)
            #remove res.orig to avoid memory leaks
            res.orig = None
            res = nres
        return res 
    
    def __getitem__(self, i): return self.layers[i]
    def append(self, l): return self.layers.append(l)
    def extend(self, l): return self.layers.extend(l)
    def insert(self, i, l): return self.layers.insert(i, l)

class MergeLayer(nn.Module):
    def __init__(self, dense:bool=False):
        super().__init__()
        self.dense = dense
    
    def forward(self, x): return torch.cat([x, x.orig], dim = 1) if self.dense else (x + x.orig)
    
def res_block(nf, dense:bool=False, norm_type=NormType.Batch, bottle:bool=False, **kwargs):
    norm2 = norm_type
    if not dense and (norm_type == NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf//2 if bottle else nf
    return SequentialEx(conv_layer(nf, nf_inner, norm_type=norm_type, **kwargs),
                        conv_layer(nf_inner, nf, norm_type=norm2, **kwargs), 
                        MergeLayer(dense))

def conv_and_res(ni, nf, self_attention=True): 
    return nn.Sequential(conv_layer(ni, nf, stride = 2), 
                         res_block(nf, bottle=True, self_attention=self_attention))
    
class MyResidualNetwork(nn.Module):
    def __init__(self, channels):
        super(MyResidualNetwork, self).__init__()
        layers = []
        for (ni, nf) in zip(channels[:-1], channels[1:]):
            layers.append(conv_and_res(ni, nf, self_attention = True))
        self.convolution = nn.Sequential(*layers)
        self.last_channel = channels[-1]
     
    def forward(self, x):
        x = self.convolution(x)
        return x
       
class SelfAttention(nn.Module):
    def __init__(self, n_channels:int):
        super().__init__()
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(tensor([0.]))
        
    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim = 1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

def conv1d(ni:int, no:int, ks:int=1, stride:int=1, padding:int=0, bias:bool=False):
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias: conv.bias.data.zero_()
    
    return spectral_norm(conv)