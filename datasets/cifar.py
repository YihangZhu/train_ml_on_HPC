from collections import Counter

import torch
import torchvision
import torchvision.datasets
from torchvision import transforms


# image size 32*32
class CIFAR(object):

    def __init__(self, logger, dataset_function, data_loader_func, data_path='./saved/data/cifar/',
                 num_classes=None,
                 imbalance_factor=None, major=None, minor=None,
                 head_class_idx=None, med_class_idx=None, tail_class_idx=None,
                 batch_size=None, num_workers=None,
                 img_max=None, imbalance_type=None, gpu_rank=None,
                 world_size=None, ddp=False
                 ):
        # if major is not None:
        #     if len(major) > 1:
        #         major = [*range(major[0], major[1] + 1)]
        #     if len(minor) > 1:
        #         minor = [*range(minor[0], minor[1] + 1)]

        if num_classes == 2:
            head_class_idx = [0]
            tail_class_idx = [1]

        self.head_class_idx = head_class_idx
        self.med_class_idx = med_class_idx
        self.tail_class_idx = tail_class_idx
        self.num_classes = num_classes
        # if config.dataset == 'cifar10':
        #     dataset_function = torchvision.datasets.CIFAR10
        # elif config.dataset == 'cifar100':
        #     dataset_function = torchvision.datasets.CIFAR100
        # else:
        #     raise Exception('Dataset name is not supported.')
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        train_mean = eval_mean = mean
        train_std = eval_std = std
        # train_set = dataset_function(data_path, train=True, download=True,
        #                              transform=transforms.Compose([transforms.ToTensor()]))
        #
        # # computer the mean and std for normalization
        # train_mean = train_set.data.mean(axis=(0, 1, 2)) / 255
        # train_std = train_set.data.std(axis=(0, 1, 2)) / 255
        # # print(train_mean, train_std)
        #
        # eval_set = dataset_function(data_path, train=False, download=False,
        #                             transform=transforms.Compose([transforms.ToTensor()]))
        #
        # eval_mean = eval_set.data.mean(axis=(0, 1, 2)) / 255
        # eval_std = eval_set.data.std(axis=(0, 1, 2)) / 255
        # print(eval_mean, eval_std)

        # **************** processing the data
        # normalize data manually:
        # https://inside-machinelearning.com/en/why-and-how-to-normalize-data-object-detection-on-image-in-pytorch-part-1/#Normalize_Data_Manually

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std),
        ])

        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(eval_mean, eval_std),
        ])

        # download the dataset
        # transform operations is not called here, it will be called when reading the image data from the dataset.
        self.train_dataset = dataset_function(root=data_path, train=True, download=True,
                                              transform=train_transform)

        self.eval_dataset = dataset_function(root=data_path, train=False, download=False,
                                             transform=eval_transform)

        self.dist_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset, num_replicas=world_size, rank=gpu_rank) if ddp else None

        self.train_dataloader = data_loader_func(
            self.train_dataset,
            batch_size=batch_size, shuffle=(self.dist_sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=self.dist_sampler
        )
        # is it the explanation why we need to use pin memory:
        # https://discuss.pytorch.org/t/when-to-set-pin-memory-to-true/19723

        # balance_sampler = ClassAwareSampler(train_dataset)
        # self.train_balance = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=batch_size, shuffle=False,
        #     num_workers=num_works, pin_memory=True, sampler=balance_sampler)

        self.eval_dataloader = data_loader_func(
            self.eval_dataset,
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        logger.info(f'==============train data==============', gpu_rank=gpu_rank)
        logger.info(self.train_dataset.class_to_idx, gpu_rank=gpu_rank)
        logger.info(self.train_dataset.data.shape, gpu_rank=gpu_rank)
        logger.info(Counter(self.train_dataset.targets), gpu_rank=gpu_rank)
        logger.info(f'==============test data==============', gpu_rank=gpu_rank)
        logger.info(self.eval_dataset.class_to_idx, gpu_rank=gpu_rank)
        logger.info(self.eval_dataset.data.shape, gpu_rank=gpu_rank)
        logger.info(Counter(self.eval_dataset.targets), gpu_rank=gpu_rank)


def cifar10(logger, **kwargs):
    # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3,
    #     # 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 10

    if kwargs['num_classes'] != 2:
        if 'imbalance_type' in kwargs and kwargs['imbalance_type'] == 'step':
            kwargs.update({
                'head_class_idx': [*range(0, 5)],
                'tail_class_idx': [*range(5, 10)]
            })
        elif 'head_class_idx' not in kwargs:
            kwargs.update({
                'head_class_idx': [*range(0, 3)],
                'med_class_idx': [*range(3, 6)],
                'tail_class_idx': [*range(6, 10)],
            })

    return CIFAR(logger=logger, dataset_function=torchvision.datasets.CIFAR10, **kwargs)


def cifar100(logger, **kwargs):
    if 'num_classes' not in kwargs:
        kwargs['num_classes'] = 100

    if kwargs['num_classes'] != 2:
        if 'imbalance_type' in kwargs and kwargs['imbalance_type'] == 'step':
            kwargs.update({
                'head_class_idx': [*range(0, 50)],
                'tail_class_idx': [*range(50, 100)]
            })
        elif 'head_class_idx' not in kwargs:
            kwargs.update({
                'head_class_idx': [*range(0, 35)],
                'med_class_idx': [*range(35, 70)],
                'tail_class_idx': [*range(70, 100)],
            })

    return CIFAR(logger=logger, dataset_function=torchvision.datasets.CIFAR100, **kwargs)
