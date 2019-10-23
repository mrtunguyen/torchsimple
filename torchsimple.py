from lib import *
from .callback import (Callback, Callbacks, ProgressBarCallback, 
                       PredictionsSaverCallback, OneCycleLR, DefaultLossCallback, DefaultMetricsCallback, 
                       Logger, LRFinder, CheckpointSaverCallback, DefaultSchedulerCallback, 
                       EarlyStoppingCallback, DefaultOptimizerCallback)
from .utils import DotDict, freeze_to, freeze, unfreeze, load_state_dict, convert_model_to_half
from .data import DataOwner
from .parallel import DataParallelCriterion, DataParallelModel

class Trainer:
    
    def __init__(self,
                 model: torch.nn.Module,
                 dataowner: DataOwner,
                 criterion: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
                 target_key: str = "label",
                 preds_key: str = "preds",
                 metrics_cb:Optional[Callback]=None,
                 metrics: Optional[Dict[str, Callable]] = None,
                 opt: Optional[Type[torch.optim.Optimizer]] = None,
                 opt_params: Optional[Dict] = None,
                 device: Optional[torch.device] = None,
                 step_fn: Optional[Callable] = None,
                 loss_cb: Optional[Callback] = None,
                 opt_cb: Optional[Callback] = None,
                 callbacks: Optional[Union[List, Callbacks]] = None) -> None:
        
        self.state = DotDict()
        self.state = DotDict()
        
        self.state.model = model
        self.state.dataowner = dataowner or DataOwner(None, None, None)
        self.target_key = target_key
        self.preds_key = preds_key
        
        self.state.criterion = criterion
        
        self.opt = opt or optim.SGD
        self.opt_params = opt_params or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state.model.to(self.device)
        
        self.step_fn = step_fn or self.default_step_fn
        
        loss_cb = loss_cb or DefaultLossCallback(target_key, preds_key)
        opt_cb = opt_cb or DefaultOptimizerCallback()
        metrics_cb = metrics_cb or DefaultMetricsCallback(self.target_key, self.preds_key, metrics)
        callbacks = callbacks or []
        self.core_callbacks =  [loss_cb,
                                metrics_cb,
                                opt_cb,
                                ProgressBarCallback()] + callbacks
        
        callbacks = self.core_callbacks[:]
        self.callbacks = Callbacks(callbacks)
        
        self.state.checkpoint = ""
        
        self.state.stop_iter = None
        #stop epoch
        self.state.stop_epoch = False
        #stop training
        self.state.stop_train = False
        #the scheduler attribute
        self.state.sched = None
        
        self.state.do_log = False
        
        self.state.preds = None
        #fp16 flag
        self.state.use_fp16 = None
        
        
    def fit(self,
           lr:float,
           epochs:int,
           skip_val:bool=False,
           opt=None,
           opt_params=None,
           sched=None,
           sched_params=None,
           stop_iter=None,
           logdir=None,
           cp_saver_params=None,
           early_stop_params=None,
           use_fp16:bool=False):
        
        if self.state.criterion is None:
            raise Exception("Needs criterion.")
        
        if self.state.dataowner.train_dl is None:
            raise Exception("Needs Dataowner.")
            
        self.state.stop_iter = stop_iter
        self.state.use_fp16 = use_fp16
        
        callbacks = self.callbacks
        
        opt = opt or self.opt
        opt_params = opt_params or self.opt_params
        params =  (p for p in self.state.model.parameters() if p.requires_grad)
        self.state.opt = opt(params=params, lr=lr, **opt_params)
        
        if self.state.use_fp16:
            self.state.model = convert_model_to_half(self.state.model)
            self.state.opt = FP16_Optimizer(self.state.opt,
                                                 static_loss_scale = 1,
                                                 dynamic_loss_scale = True,
                                                 verbose=False)
        
        if sched:
            sched_params = sched_params or {}
            self.state.sched = sched(optimizer=self.state.opt, **sched_params)
            sched_cb = DefaultSchedulerCallback(sched=self.state.sched)
            self.callbacks = Callbacks(self.callbacks.callbacks + [sched_cb])
        
        if logdir:
            self.state.do_log = True
            self.state.metrics = defaultdict(dict)
            tboard_cb = Logger(logdir)
            self.callbacks = Callbacks(self.callbacks.callbacks + [tboard_cb])
            
        cp_saver_params = cp_saver_params or {}
        if cp_saver_params:
            cp_saver_cb = CheckpointSaverCallback(**cp_saver_params)
            self.callbacks = Callbacks(self.callbacks.callbacks + [cp_saver_cb])
            
        early_stop_params = early_stop_params or {}
        if early_stop_params:
            early_stop_cb = EarlyStoppingCallback(**early_stop_params)
            self.callbacks = Callbacks(self.callbacks.callbacks + [early_stop_cb])
            
        try:
            self.callbacks.on_train_begin(self.state)
            
            for epoch in range(epochs):
                self.set_mode("train")
                self._run_epoch(epoch, epochs)
                
                if not skip_val:
                    self.set_mode("val")
                    self._run_epoch(epoch, epochs)
                    
                if self.state.stop_train:
                    self.state.stop_train = False
                    print(f"Early stopped on {epoch + 1} epoch")
                    break
            
            self.callbacks.on_train_end(self.state)
        
        finally:
            self.state.pbar.close()
            self.state.do_log = False
            self.callbacks = callbacks
            
    def fit_one_cycle(self,
                      max_lr:float,
                      cycle_len:int,
                      momentum_range=(0.95, 0.85),
                      div_factor:float=25,
                      increase_fraction:float=0.3,
                      opt=None,
                      opt_params=None,
                      stop_iter = None,
                      skip_val = None,
                      logdir=None,
                      cp_saver_params=None,
                      early_stop_params=None,
                      use_fp16=False):
        
        callbacks = self.callbacks
        
        one_cycle_cb = OneCycleLR(max_lr, momentum_range, div_factor, increase_fraction,
                                    final_div=None, total_epochs=cycle_len, start_epoch=None)
        try:
            self.callbacks = Callbacks(callbacks.callbacks + [one_cycle_cb])
            
            self.fit(lr=max_lr,
                     epochs = cycle_len,
                     opt=opt,
                     opt_params = opt_params,
                     logdir = logdir,
                     skip_val = skip_val,
                     stop_iter = stop_iter,
                     cp_saver_params = cp_saver_params,
                     early_stop_params = early_stop_params,
                     use_fp16 = use_fp16)
            
        finally:
            #set old callbacks without OneCycleCallback
            self.callbacks = callbacks
            
    def find_lr(self, 
                final_lr:float,
                logdir=None,
                init_lr:float=1e-6,
                n_steps=None,
                opt=None,
                opt_params=None):
        
        if logdir:
            Path(logdir).mkdir(exist_ok=True)
        
        len_loader = len(self.state.dataowner.train_dl)
        n_steps = n_steps if n_steps is not None else len_loader
        n_epochs = max(1, int(np.ceil(n_steps /len_loader)))
        
        callbacks = self.callbacks
        try: 
            lr_finder_cb = LRFinder(final_lr=final_lr,
                                    init_lr=init_lr,
                                    n_steps=n_steps)
            self.callbacks = Callbacks(self.core_callbacks + [lr_finder_cb])
            self.fit(lr=init_lr, epochs = n_epochs, skip_val=True, logdir=logdir,
                     opt=opt, opt_params=opt_params)
        finally:
            self.callbacks = callbacks
        
    def _run_epoch(self, 
                   epoch:int,
                   epochs:int):
        
        self.callbacks.on_epoch_begin(epoch, epochs, self.state)
        
        with torch.set_grad_enabled(self.is_train):
            for i, batch in enumerate(self.state.loader):
                self.state.batch = self.to_device(batch)
                self.callbacks.on_batch_begin(i, self.state)
                self.state.out = self.step()
                self.callbacks.on_batch_end(i, self.state)
                if self.state.checkpoint:
                    self.save(self.state.checkpoint)
                    self.state.checkpoint = ""
                
                if (self.state.stop_iter and 
                    self.state.mode == "train" and 
                    i == self.state.stop_iter - 1):
                    #break if in train mode and early stop is set
                    self.state.stop_epoch = True
                    
                if self.state.stop_epoch:
                    self.state.stop_epoch = False
                    break
        self.callbacks.on_epoch_end(epoch, self.state)
        
        if self.state.checkpoint:
            self.save(self.state.checkpoint)
            self.state.checkpoint = ""
    
    @staticmethod
    def default_step_fn(model: torch.nn.Module,
                        batch: torch.Tensor) -> torch.Tensor:
        """Determine what your model will do with your data.

        Args:
            model: the pytorch module to pass input in
            batch: the batch of data from the DataLoader

        Returns:
            The models forward pass results
        """
        inp = batch["image"]
        return model(inp)
    
    def step(self):
        """The step function that calls each iteration
        Wraps the self.step_fn to provide a dict of predictions
        """
        preds = self.step_fn(self.state)
        
        return {self.preds_key:preds}
    
    def predict(self, savepath: Union[str, Path]) -> None:
        """Infer the model on test dataloader and saves prediction as numpy array

        Args:
            savepath: the directory to save predictions
        """
    
        return self.predict_loader(loader = self.state.dataowner.test_dl,
                                   savepath = savepath)
    def predict_loader(self,
                       loader: DataLoader,
                       savepath: Union[str, Path],
                       preds_cb: Optional[Callable]=None) -> None:
        """Infer the model on dataloader and saves prediction as numpy array

        Args:
            loader: the dataloader for generating predictions
            savepath: the directory to save predictions
        """
        callbacks = self.callbacks
        
        if preds_cb:
            tmp_callbacks = Callbacks([ProgressBarCallback(),
                                       preds_cb])
        else:
            tmp_callbacks = Callbacks([ProgressBarCallback(),
                                       PredictionsSaverCallback(savepath,
                                                                self.preds_key)])

        self.callbacks = tmp_callbacks

        self.state.mode = "test"
        self.state.loader = loader
        self.state.model.eval()
        with torch.set_grad_enabled(False):
            self._run_epoch(1, 1)

        self.callbacks = callbacks
        preds = self.state.preds
        self.state.preds = None
        return preds
        
    def predict_tensor(self,
                       tensor: torch.Tensor,
                       to_numpy: bool = False) -> Union[torch.Tensor,
                                                        np.ndarray]:
        """Infer the model on one torch Tensor.

        Args:
            tensor: torch tensor to predict on.
                Should has [batch_size, *(one_sample_shape)] shape
            to_numpy: if True, converts predictions to numpy array

        Returns:
            Predictions on input tensor.
        """
        tensor = tensor.to(self.device)
        with torch.set_grad_enabled(False):
            self.set_mode("test")
            preds = self.state.model(tensor)
        if to_numpy:
            preds = preds.cpu().numpy()
        return preds
    
    def predict_array(self,
                      array: np.ndarray,
                      to_numpy: bool = False) -> Union[Type[torch.Tensor],
                                                       np.ndarray]:
        """Infer the model on one numpy array.

        Args:
            array: numpy array to predict on.
                Should has [batch_size, *(one_sample_shape)] shape
            to_numpy: if True, converts predictions to numpy array

        Returns:
            Predictions on input tensor.
        """
        tensor = torch.from_numpy(array)
        return self.predict_tensor(tensor, to_numpy)
    
    def predict_inputs(self,
                       inputs:Union[torch.Tensor, Collection[torch.Tensor]]):
        inputs = inputs.to(self.device)
        with torch.set_grad_enabled(False):
            self.set_mode("test")
            preds = self.state.model(*inputs)
        return preds
    
    def TTA(self,
            loader: DataLoader,
            tfms: Union[List, Dict],
            savedir: Union[str, Path],
            prefix: str = "preds") -> None:
        """Conduct the test-time augmentations procedure.

        Create predictions for each set of provided transformations and saves
        each prediction in savedir as a numpy arrays.

        Args:
            loader: loader to predict
            tfms: the list with torchvision.transforms or
                  the dict with {"name": torchvision.transforms} pairs.
                  List indexes or dict keys will be used for generating
                  predictions names.
            savedir: the directory to save predictions
            prefix: the prefix for predictions files names
        """
        if isinstance(tfms, dict):
            names = [f"{prefix}_{k}.npy" for k in tfms]
            tfms = tfms.values()
        elif isinstance(tfms, list):
            names = [f"{prefix}_{i}.npy" for i in range(len(tfms))]
        else:
            raise ValueError(f"Transforms should be List or Dict, "
                             f"got {type(tfms)}")

        default_tfms = loader.dataset.transforms
        for name, tfm in zip(names, tfms):
            loader.dataset.transforms = tfm
            savepath = Path(savedir) / name
            self.predict_loader(loader, savepath)
        loader.dataset.transforms = default_tfms
        
    def save(self, savepath: Union[str, Path]) -> None:
        """Save models state dict on the specified path.

        Args:
            savepath: the path to save the state dict.
        """
        savepath = Path(savepath)
        savepath.parent.mkdir(exist_ok=True)
        torch.save(self.state.model.state_dict(), savepath)

    def load(self,
             loadpath: Union[str, Path],
             skip_wrong_shape: bool = False) -> None:
        """Loads models state dict from the specified path.

        Args:
            loadpath: the path from which the state dict will be loaded.
            skip_wrong_shape: If False, will raise an exception if checkpoints
                weigths shape doesn't match models weights shape.
                If True, will skip unmatched weights and load only matched.
        """
        loadpath = Path(loadpath)
        checkpoint = torch.load(loadpath,
                                map_location=lambda storage, loc: storage)

        # workaround DataParallelModel
        if not isinstance(self.state.model, DataParallelModel) \
                and "module." in list(checkpoint.keys())[0]:
            # [7:] is to skip 'module.' in group name
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}

        load_state_dict(model=self.state.model,
                        state_dict=checkpoint,
                        skip_wrong_shape=skip_wrong_shape)
        return self
        
    def to_device(self,
                  batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Moves tensors in batch to self.device.

        Args:
            batch: the batch dict.

        Returns:
            The batch dict with tensors on self.device.
        """
        res = {}
        for k, v in batch.items():
            if isinstance(v, dict):
                v = self.to_device(v)
            else:
                if hasattr(v, "to"):
                    v = v.to(self.device)
                    
            res[k] = v
        return res
        
    def to_fp16(self):
        self.state.model = amp.initialize(self.state.model,
                                               opt_level = "01",
                                               verbosity = 0)
        self.state.use_fp16 = True
        return self
        
    def set_mode(self, mode: str) -> None:
        """Set the model to train or val and switch dataloaders

        Args:
            mode: 'train', 'val' or 'test', the mode of training procedure.
        """
        if mode == "train":
            self.state.model.train()
            self.state.loader = self.state.dataowner.train_dl
        elif mode == "val":
            self.state.model.eval()
            self.state.loader = self.state.dataowner.val_dl
        elif mode == "test":
            self.state.model.eval()
            self.state.loader = self.state.dataowner.test_dl
        self.state.mode = mode

    def freeze_to(self,
                  n: int,
                  freeze_bn: bool = False,
                  model_attr: Optional[str] = None) -> None:
        """Freeze model or model's part till specified layer.

        Args:
            n: the layer number to freeze to
            freeze_bn: if True batchnorm layers will be frozen too
            model_attr: the name of the model attribute if you want to specify
                when you want to freeze layers.
                For examples see example ipynb's.
        """

        module = self.get_model_attr(model_attr)
        freeze_to(module, n, freeze_bn)

    def freeze(self,
               freeze_bn: bool = False,
               model_attr: Optional[str] = None) -> None:
        """Freeze model or model's part till the last layer

        Args:
            freeze_bn: if True batchnorm layers will be frozen too
            model_attr: the name of the model attribute if you want to specify
                when you want to freeze layers.
                For examples see example ipynb's.
        """
        module = self.get_model_attr(model_attr)
        freeze(module, freeze_bn)

    def unfreeze(self,
                 model_attr: Optional[str] = None) -> None:
        """Unfreeze all model or model's part layers.

        Args:
            model_attr: the name of the model attribute if you want to specify
                when you want to freeze layers.
                For examples see example ipynb's.
        """
        module = self.get_model_attr(model_attr)
        unfreeze(module)

    def get_model_attr(self, model_attr: Union[str, None]) -> torch.nn.Module:
        """Get models attribute by name or return the model if name is None.

        Args:
            model_attr: models attribute name to get. If none, than the model
                will be returned.

        Returns:
            The models attribute or the model itself.
        """
        if self.state.parallel:
            model = self.state.model.module
        else:
            model = self.state.model

        if model_attr is not None:
            module = getattr(model, model_attr)
        else:
            module = model
        return module

    def add_callbacks(self, callbacks: List[Callback]) -> None:
        """Add callbacks to the beginning of self.callbacks"""
        self.callbacks = Callbacks(callbacks + self.callbacks.callbacks)
        
    @property
    def is_train(self):
        return self.state.mode == "train"