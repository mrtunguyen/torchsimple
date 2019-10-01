from typing import List
from .callback import Callback
from torchsimple.utils import DotDict

__all__ = ['DebuggerCallback']

class DebuggerCallback(Callback):
    def __init__(self, when: List[str], modes: List[str]) -> None:
        self.when = when
        self.modes = modes

    def on_batch_begin(self, i: int, state: DotDict) -> None:
        if "on_batch_begin" in self.when:
            if state.core.mode == "train" and "train" in self.modes:
                set_trace()
            if state.core.mode == "val" and "val" in self.modes:
                set_trace()
            if state.core.mode == "test" and "test" in self.modes:
                set_trace()

    def on_batch_end(self, i: int, state: DotDict) -> None:
        if "on_batch_end" in self.when:
            if state.core.mode == "train" and "train" in self.modes:
                set_trace()
            if state.core.mode == "val" and "val" in self.modes:
                set_trace()
            if state.core.mode == "test" and "test" in self.modes:
                set_trace()

    def on_epoch_begin(self, epoch: int, epochs: int, state: DotDict) -> None:
        if "on_epoch_begin" in self.when:
            if state.core.mode == "train" and "train" in self.modes:
                set_trace()
            if state.core.mode == "val" and "val" in self.modes:
                set_trace()
            if state.core.mode == "test" and "test" in self.modes:
                set_trace()

    def on_epoch_end(self, epoch: int, state: DotDict) -> None:
        if "on_epoch_end" in self.when:
            if state.core.mode == "train" and "train" in self.modes:
                set_trace()
            if state.core.mode == "val" and "val" in self.modes:
                set_trace()
            if state.core.mode == "test" and "test" in self.modes:
                set_trace()

    def on_train_begin(self, state: DotDict) -> None:
        if "on_train_begin" in self.when:
            set_trace()

    def on_train_end(self, state: DotDict) -> None:
        if "on_train_end" in self.when:
            set_trace()