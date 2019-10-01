import torch
from torch import nn
from torchvision import models

pretrained_path_dict = {'resnet101' : '/mnt/thanhtu/pytorch_pretrained_model/resnet101-5d3b4d8f.pth',
                        'resnet18' : '/mnt/thanhtu/pytorch_pretrained_model/resnet18-5c106cde.pth',
                        'resnet34' : '/mnt/thanhtu/pytorch_pretrained_model/resnet34-333f7ec4.pth',
                        'resnet50' : '/mnt/thanhtu/pytorch_pretrained_model/resnet50-19c8e357.pth',
                        'vgg16' : '/mnt/thanhtu/pytorch_pretrained_model/vgg16-397923af.pth',
                        'vgg19' : '/mnt/thanhtu/pytorch_pretrained_model/vgg19-dcbb9e9d.pth'}

__all__ = ['pretrained_models']

def pretrained_models(name_model, pretrained=True):
    pretrain_model = getattr(models, name_model)()
    if pretrained:
        try:
            pretrained_path = pretrained_path_dict[name_model]
        except:
            raise Exception(f'{name_model} is not in our base pretrained model')
        pretrained_dict = torch.load(pretrained_path)
        model_dict = pretrain_model.state_dict()
        model_dict.update(pretrained_dict)
        pretrain_model.load_state_dict(model_dict)
    
    return pretrain_model