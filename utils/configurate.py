import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.distributed import init_process_group, destroy_process_group

from utils.utils import load_yaml, Logging, set_reproducibility, get_free_port, MultiEpochsDataLoader


class ConfigParser:
    def __init__(self, args):
        args = _check_args(args)
        set_reproducibility(args.seed)

        # get total number of GPUs at the node
        if args.port is None:
            args.port = get_free_port()
        self._config = load_yaml('utils/menu.yaml')
        self._load_config(args.cfg)
        if args.cfg_specify is not None:
            self._load_config(args.cfg_specify)

        self._config['trainer']['name'] = os.path.basename(args.cfg).split('.')[0]
        if args.cfg_specify is not None:
            string = os.path.basename(args.cfg_specify).split('.')[0].split('_')[-1]
            self._config['trainer']['name'] += f"_{string}"

        self._config['trainer']['mode'] = args.mode
        self._config['trainer']['resume'] = args.resume
        self._config['trainer']['checkpoint'] = args.checkpoint
        self._config['trainer']['seed'] = args.seed
        self._config['trainer']['print_freq'] = args.print_freq
        self._config['dataset']['args']['num_workers'] = args.wks
        self._config['dataset']['args']['batch_size'] = self._config['dataset']['args']['total_batch_size']
        self._setup_ddp(args)
        _clear_config(self._config)

        # =================setup folders===========
        dirs = dict()
        time_str = time.strftime('%Y-%m%d-%H%M')
        dirs['save_path'] = Path(f"saved/"
                                 f"{self._config['trainer']['mode']}/"
                                 f"{time_str}_{self._config['trainer']['name']}_nodes{args.nodes}")
        if self.config['trainer']['mode'] in ['test', 'train']:
            dirs['model_dir'] = dirs['save_path'] / Path('ckps')
            dirs['model_dir'].mkdir(parents=True, exist_ok=True)
        self._dirs = dirs

        # ============ setup logger
        # dirs, log_name = prepare(config, args)
        # when running with multiprocessing, only log for process 0 of node args.nr
        logger = Logging(dirs['save_path'], 'log_file')
        self.logger = logger
        logger.info(f"Folder created: {dirs['save_path']}", gpu_rank=self._config['ddp']['node_rank'])
        logger.info(self, gpu_rank=self._config['ddp']['node_rank'])
        del self._config['dataset']['args']['total_batch_size']
        gpus_per_node = torch.cuda.device_count()
        logger.info(f"Node {self._config['ddp']['node_rank']} has {gpus_per_node} GPUs")

    def __str__(self):
        string = ""
        return self.__get_str_config(self._config, string, horizon="", first=True)

    def __get_str_config(self, item, string, horizon, first=False):
        if not first:
            horizon += "\t"
            string += '\n'
        for k, v in item.items():
            string += f"{horizon}{k}:"
            if isinstance(v, dict):
                string = self.__get_str_config(v, string, horizon)
            else:
                string += f' {v}\n'
        return string

    def _setup_ddp(self, args):
        num_unused_gpus = args.nodes * args.gpus - args.world_size
        if args.world_size > 0 and num_unused_gpus >= args.gpus:
            raise Exception(f"Too many nodes are booked: "
                            f"{args.nodes} nodes * {args.gpus} gpus >> {args.world_size} world_size.")

        ddp = self._config['ddp']
        # =================setup hardware environment
        # assert args.gpus == torch.cuda.device_count()
        # the total number of GPUs to run considering all the applied nodes,
        # assuming each node has the same number of GPUs
        ddp['world_size'] = args.world_size
        ddp['num_nodes'] = args.nodes
        ddp['num_gpus_per_node'] = [args.gpus for _ in range(args.nodes)]
        ddp['num_gpus_per_node'][-1] -= num_unused_gpus

        if 'SLURM_PROCID' in os.environ:
            ddp['node_rank'] = int(os.environ['SLURM_PROCID'])
        else:
            ddp['node_rank'] = args.nr

        # the batch will be divided by all the available GPUs
        if ddp['world_size'] > 1:
            ddp['on'] = True
            self._config['dataset']['args']['batch_size'] = int(
                self._config['dataset']['args']['batch_size'] / ddp['world_size'])
            if args.ip is not None and args.port is not None:
                ddp['dist_url'] = f'tcp://{args.ip}:{args.port}'
            else:
                if 'MASTER_ADDR' not in os.environ:
                    raise Exception('Master address is not set.')
                if 'MASTER_PORT' not in os.environ:
                    raise Exception('Master port is not set.')

        # if ddp['num_gpus_per_node'] > 1:
        #     args.wks = int(np.floor(args.wks / ddp['num_gpus_per_node']))

    def init_obj(self, obj, module, *args, allow_override=False, device=None, gpu_id=None, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self.config[obj]['name']
        module_args = dict(self.config[obj]['args']) if 'args' in self.config[obj] else dict()
        if not allow_override:
            assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        module = getattr(module, module_name)(*args, **module_args)

        if device is not None:
            module.to(device)
            if self.config['ddp']['on'] and obj == 'model':
                module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[gpu_id])

        return module

    def _load_config(self, cfg_file):
        msg_no_key = 'key is not found in the template file'
        config = load_yaml(cfg_file)
        self.__load_config(self._config, config, msg_no_key)

    def __load_config(self, template, config, msg_no_key):
        for k, v in config.items():
            assert k in template.keys(), f'{msg_no_key}: {k}'
            if isinstance(template[k], dict):
                self.__load_config(template[k], config[k], msg_no_key)
            else:
                template[k] = config[k]

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def dirs(self):
        return self._dirs

    def init_ddp(self, gpu_id):
        gpu_rank = gpu_id
        ddp = self.config['ddp']
        data_loader_func = MultiEpochsDataLoader
        if ddp['on']:  # initialize torch distributed parallel package for running with multiple nodes.
            # This is the global rank of the process within all the processes (one process per GPU)
            # gpus_ranks includes the rank for all the gpus
            count = 0
            gpu_ranks = [[0 for _ in range(ddp['num_gpus_per_node'][node_id])] for node_id in range(ddp['num_nodes'])]
            for node_id in range(ddp['num_nodes']):
                for g_id in range(ddp['num_gpus_per_node'][node_id]):
                    gpu_ranks[node_id][g_id] = count
                    count += 1
            self.logger.info(f"Ranks for all the applied GPUs: {gpu_ranks}", gpu_rank)
            gpu_rank = gpu_ranks[ddp['node_rank']][gpu_id]
            self.logger.info(f"Initializing distributed package for GPU {gpu_rank}/{ddp['world_size']} "
                             f"(node {ddp['node_rank']}, GPU {gpu_id}/{ddp['num_gpus_per_node'][ddp['node_rank']]}) "
                             f"with {ddp['dist_url']}.....",
                             gpu_rank=0)  # print out this for all GPUs
            init_process_group(backend=ddp['dist_backend'], init_method=ddp['dist_url'],
                               world_size=ddp['world_size'], rank=gpu_rank)
            self.logger.info(f"Distributed package is initialize for GPU {gpu_rank}/{ddp['world_size']}", gpu_rank=0)

            device = f'cuda:{gpu_id}'
            torch.cuda.set_device(gpu_id)
        else:
            if torch.cuda.is_available():
                device = f'cuda:0'
                self.logger.info('===> using one GPU for training', gpu_rank=gpu_id)
            else:
                device = 'cpu'
                self.logger.info('===> using CPU', gpu_rank=gpu_id)
                data_loader_func = torch.utils.data.DataLoader

        return data_loader_func, device, gpu_rank

    def stop_ddp(self):
        if self.config['ddp']['on']:
            destroy_process_group()

    def output(self, gpu_rank):
        """only output at rank 0 GPU out of all the available GPUs"""
        return not self.config['ddp']['on'] or (self.config['ddp']['on'] and gpu_rank == 0)


def _clear_config(config):
    # clean up the unused taps in the template
    for k in list(config.keys()):
        if isinstance(config[k], dict):
            _clear_config(config[k])
        else:
            if config[k] is None:
                del config[k]


def _check_args(args):
    if args.nodes == 1:
        args.nr = 0
    if args.mode in ['validate', 'visualize', 'test-sample'] or args.resume:
        if args.checkpoint is None:
            raise Exception('Please setup path for loading checkpoints.')
    else:
        args.checkpoint = None
    if not args.deterministic:
        args.seed = np.random.randint(1e8)

    return args
