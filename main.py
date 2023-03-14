import argparse
import time

import torch
import torch.multiprocessing as mp
import torch.optim
import torch.utils.data

import datasets
import models
from utils import *


def main(args):
    config = ConfigParser(args)
    if config.config['ddp']['on']:
        # Use torch.multiprocessing.spawn to launch distributed processes:
        # the main_worker process function
        num_gpus = config.config['ddp']['num_gpus_per_node'][config.config['ddp']['node_rank']]
        mp.spawn(main_worker, nprocs=num_gpus, args=(config,))
    else:
        # Simply call main_worker function
        main_worker(None, config)


def main_worker(gpu_id, config):
    """gpu_id is the id per node, while gpu_rank is the gpu rank among all the available gpus in the nodes"""
    logger = config.logger
    start_time = time.time()

    data_loader_func, device, gpu_rank = config.init_ddp(gpu_id)

    dataset = config.init_obj('dataset', datasets, logger=config.logger, data_loader_func=data_loader_func,
                              gpu_rank=gpu_rank,
                              world_size=config.config['ddp']['world_size'],
                              ddp=config.config['ddp']['on'])

    learning_model = config.init_obj('model', models, device=device, gpu_id=gpu_id, num_classes=dataset.num_classes)

    optimizer = config.init_obj('optimizer', torch.optim, learning_model.parameters())

    num_epochs = config.config['trainer']['num_epochs']
    lr_scheduler = config.init_obj('lr_scheduler', lr_schedulers, optimizer,
                                   lr=config.config['optimizer']['args']['lr'],
                                   num_epochs=num_epochs)

    criterion = config.init_obj('loss', models, device=device, dataset=dataset)
    trainer_ = Trainer(start_time, gpu_rank=gpu_rank, device=device)
    trainer_param = config.config['trainer']
    if 'checkpoint' in trainer_param:
        try:
            trainer_.load_checkpoint(learning_model, optimizer, trainer_param['checkpoint'], device, logger, gpu_rank,
                                     config.config['ddp']['on'])
        except FileNotFoundError:
            exit(-1)

    train_errors, validate_errors, norm_head_med_tail = trainer_.train_model(
        learning_model, criterion, optimizer, dataset, lr_scheduler, num_epochs, logger, config
    )
    if config.output(gpu_rank):
        plot_errors(train_errors, validate_errors, config.dirs['save_path'] / 'error.png')

    logger.info(f"training completed, time: {round(time.time() - start_time)}s", gpu_rank=gpu_rank)
    config.stop_ddp()
    # del train_errors, validate_errors, learning_model, dataset, train_loader, val_loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    # ============= for trainer
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='config/cifar/Res32_cifar10_1.yaml',
                        type=str)
    parser.add_argument('--cfg_specify',
                        help='experiment configure file name',
                        # default='config/cifar/specify/cifar10_batch8.yaml'
                        type=str)
    parser.add_argument('--resume',
                        help='whether resume training from a checkpoint.',
                        default=False,
                        type=bool)
    parser.add_argument('--checkpoint',
                        type=str,
                        help='the path for the checkpoint to be loaded.')
    parser.add_argument('--mode',
                        help='options: test, validate, visualize',
                        default='test',
                        type=str)
    parser.add_argument('--deterministic',
                        help='fix random seed?',
                        default=True,
                        type=bool)
    parser.add_argument('--seed',
                        help='random seed',
                        default=0,
                        type=int)
    parser.add_argument('--print_freq',
                        help='the frequency of recording the training log.',
                        default=40,
                        type=int)

    # ============= for ddp
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='the total number of nodes we’re going to use')
    parser.add_argument('--gpus', default=0, type=int,
                        help='the total number of gpus available on each node')
    parser.add_argument('--world_size', default=0, type=int,
                        help='the total number of gpus we need to run the experiment, '
                             'e.g., node 0 2 gpus, node 1 3 gpus, world_size=5')
    parser.add_argument('--nr', default=None, type=int,
                        help='the rank of the current node within all the nodes, and goes from 0 to args.nodes-1')
    parser.add_argument('--ip', default=None, help='the ip address for MASTER_ADDR')
    parser.add_argument('--port', default=None, help='the free port, set arbitrarily as long as it is free', )
    parser.add_argument('--wks', default=2, type=int,
                        help='number of workers for each GPU, this ideally should be around 15, '
                             'too large or too small will make the system inefficient'
                             'https://chtalhaanwar.medium.com/pytorch-num-workers-a-tip-for-speedy-training'
                             '-ed127d825db7#:~:text=Theoretically%2C%20greater%20the%20num_workers%2C%20more,'
                             'performance%20start%20diminishing%20beyond%20that.')

    main(parser.parse_args())
