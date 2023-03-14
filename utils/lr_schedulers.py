import math


class LR_scheduler:
    def __init__(self, optimizer, name, num_epochs, lr, warmup=False):
        self.optimizer = optimizer
        self.lr = lr
        self.num_epochs = num_epochs
        self.do_warmup = warmup
        if name == 'linear':
            self.update_func = self.adjust_learning_rate_linear
        elif name == 'cosine':
            self.update_func = self.adjust_learning_rate_cosine

    def update_lr(self, epoch):
        new_lr = self.update_func(epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        return new_lr

    def adjust_learning_rate_linear(self, epoch):
        """Sets the learning rate"""
        epoch = epoch + 1
        if epoch < 5:
            lr = self.warmup(epoch)
        elif epoch >= 0.75 * self.num_epochs:
            lr = self.lr * 0.01
        elif epoch >= 0.5 * self.num_epochs:
            lr = self.lr * 0.1
        else:
            lr = self.lr
        return lr

    def warmup(self, epoch):
        if self.do_warmup:
            return self.lr * (epoch / 5)
        else:
            return self.lr

    def adjust_learning_rate_cosine(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch < 5:
            lr = self.warmup(epoch)
        else:
            lr_min = 0
            lr_max = self.lr
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(epoch / self.num_epochs * 3.1415926))
        return lr


def linear(*args, **kwargs):
    return LR_scheduler(*args, name='linear', **kwargs)


def cosine(*args, **kwargs):
    return LR_scheduler(*args, name='cosine', **kwargs)
