import os
import random
from pathlib import Path

import numpy as np
import torch
from ruamel.yaml import YAML


def set_reproducibility(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_logger(file_dir, file_name):
    print('=> creating {}'.format(file_dir))
    file_dir.mkdir(parents=True, exist_ok=True)
    log_file = '{}.txt'.format(file_name)
    final_log_file = file_dir / log_file
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)
    return logger


# logging in this way when using multiprocessing package.
class Logging:
    def __init__(self, file_dir, file_name):
        file_dir.mkdir(parents=True, exist_ok=True)
        log_file = '{}.txt'.format(file_name)
        self.log_file = file_dir / log_file

    def info(self, message, gpu_rank=0, console=True):
        # only log rank 0 GPU if running with multiple GPUs/multiple nodes.
        if gpu_rank is None or gpu_rank == 0:
            if console:
                print(message)

            with open(self.log_file, 'a') as f:  # a for append to the end of the file.
                print(message, file=f)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def get_free_port():
    import socket
    sock = socket.socket()
    sock.bind(('', 0))
    free_port = sock.getsockname()[1]
    return free_port


def save_yaml(file, path):
    with open(path, 'wb') as f:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.dump(file, f)


def load_yaml(path):
    with open(path, 'rb') as f:
        yaml = YAML()
        dt = yaml.load(f)
        return dt


def check_file(file_path):
    path = Path(file_path)
    return path.is_file()


def check_dir(dir_path):
    path = Path(dir_path)
    return path.is_dir()


def get_current_dir():
    os.getcwd()
