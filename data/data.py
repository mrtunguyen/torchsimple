from torchsimple.lib import *

class MyDataset(Dataset):
    """The easiest way to store the dataset info is pandas DataFrame.
    So MyDataset gets a DataFrame as a source of data"""
    def __init__(self,
                 df: pd.DataFrame,
                 reader_fn: Callable,
                 transforms: Optional[Compose] = None) -> None:
        # transform df to list
        self.data = list(df.iterrows())  # for multiprocessing
        self.reader_fn = reader_fn
        self.transforms = transforms

    def __getitem__(self, ind: int) -> Dict:
        datum = self.reader_fn(*self.data[ind])
        if self.transforms is not None:
            datum = self.transforms(datum)
        return datum

    def __len__(self) -> int:
        return len(self.data)


DataOwner = namedtuple("DataOwner", ["train_dl", "train_ds", "val_dl", "val_ds", "test_dl", "test_ds"])
