import os
import sys
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import datetime
from tqdm import tqdm, tqdm_notebook
from pandas_profiling import ProfileReport
from pathlib import Path
from pdb import set_trace

#image
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import PIL
from PIL import Image, ImageOps

#basic libs
import math
import re
import warnings
from typing import Any, Callable, List, Tuple, Type, Dict, Union, Optional, Collection
from collections import namedtuple, defaultdict, deque
import collections

#sklearn
from sklearn.pipeline import make_pipeline, make_union, Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, StratifiedShuffleSplit, 
                                     GridSearchCV, RandomizedSearchCV, cross_val_score)
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (mean_absolute_error, mean_squared_error, log_loss, 
                             confusion_matrix, classification_report)
from category_encoders import OrdinalEncoder, TargetEncoder

# torch 
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler, Sampler, BatchSampler
from torch.nn.parallel.data_parallel import data_parallel
from torch.nn.utils.rnn import *
from torch.nn.parallel import DistributedDataParallel

#torchvision
from torchvision import transforms
from torchvision.transforms import Compose, Normalize
from torchvision.models import *
from torchvision.models.segmentation import (DeepLabV3, deeplabv3_resnet101, deeplabv3_resnet50,
                                             FCN, fcn_resnet101)
from torchvision.models.detection import (FasterRCNN, fasterrcnn_resnet50_fpn, FastRCNNPredictor, 
                                          RegionProposalNetwork)

#apex - mixed precision
from apex.fp16_utils import FP16_Optimizer
from apex.parallel import DistributedDataParallel
