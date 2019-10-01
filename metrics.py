from callback import Callback


class MetricCallback(Callback):
    def on_epoch_begin(self, **kwargs):
        self.targets, self.preds = Tensor([]), Tensor([])
        
    def on_batch_end(self, last_output:Tensor, last_target:Tensor)