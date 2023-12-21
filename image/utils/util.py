import numpy as np
import torch
import glob
import json

from itertools import repeat
from collections import OrderedDict
from pathlib import Path


def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader


def set_seed(seed=0):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def get_all_files(pattern):
    file_paths = glob.glob(pattern)

    if len(file_paths) < 1:
        return None
    else:
        return file_paths

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(file_path):
    file_path = Path(file_path)
    with file_path.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)