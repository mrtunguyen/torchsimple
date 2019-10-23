from .callback import Callback


def dice_coef(pred: torch.Tensor, 
              target: torch.Tensor, 
              eps:float=1e-9,
              return_iou:bool=False):
    
    batch_size = pred.shape[0]
    pred = pred.view(batch_size, -1)
    target = target.view(batch_size, -1)
    intersection = (pred * true).sum(dim=1).float()
    union = (pred + true).sum(dim=1).float()
    
    if not iou: res = (2. * intersection + eps) / (union + eps)
    else: res = intersection / (union - intersection + eps)
    res[union == 0.] = 1.
    
    return res.mean()