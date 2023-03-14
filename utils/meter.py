import numpy as np
import torch
import torch.nn.functional as F


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def __repr__(self):
        return self.__str__()


def _get_batch_fmtstr(num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = _get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, logger, gpu_id):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries), gpu_rank=gpu_id)


class ImbalanceAccuracy:
    def __init__(self, dataset, device):
        self.num_classes = dataset.num_classes
        self.head_class_idx = dataset.head_class_idx
        self.med_class_idx = dataset.med_class_idx
        self.tail_class_idx = dataset.tail_class_idx

        # total number of predicated correctly objects in each class
        self.true_pos = torch.zeros(self.num_classes).to(device)
        self.false_pos = torch.zeros(self.num_classes).to(device)
        self.false_neg = torch.zeros(self.num_classes).to(device)
        # true positive + false negative equals total number of objects in each class regarding the instances loaded

    def update(self, target, output):
        _, predicted = output.max(1)  # return the index of the maximum value at dimension 1.
        target_one_hot = F.one_hot(target, self.num_classes)  # B x num_class matrix.
        predict_one_hot = F.one_hot(predicted, self.num_classes)  # B x num_class matrix

        self.true_pos += (target_one_hot + predict_one_hot == 2).sum(dim=0).to(torch.float)
        self.false_pos += (predict_one_hot - target_one_hot == 1).sum(dim=0).to(torch.float)
        self.false_neg += (predict_one_hot - target_one_hot == -1).sum(dim=0).to(torch.float)

    def calculate(self, logger, gpu_rank):
        precision_classes = self.true_pos / (self.true_pos + self.false_pos)
        recall_classes = self.true_pos / (self.true_pos + self.false_neg)

        report = self._calculate(recall_classes, 'acc', "[recall] ")
        report += self._calculate(precision_classes, 'pcs', "[precision] ")

        logger.info(report, gpu_rank=gpu_rank)
        return report

    def _calculate(self, metric_values, sub_prefix, prefix=''):
        metric_values = check_nan(metric_values)
        report = f'{prefix}'
        if self.head_class_idx is not None:
            head = metric_values[self.head_class_idx].mean() * 100
            report += f"{sub_prefix}@head: {head:.3f}\t"
        if self.med_class_idx is not None:
            med = metric_values[self.med_class_idx].mean() * 100
            report += f"{sub_prefix}@med: {med:.3f}\t"
        if self.tail_class_idx is not None:
            tail = metric_values[self.tail_class_idx].mean() * 100
            report += f"{sub_prefix}@tail: {tail:.3f}\t"

        ave_acc_classes = metric_values.mean() * 100  # average accuracy across all the classes. 1/C\sum_{i}n(i,i)/N_i
        report += f"{sub_prefix}@class: {ave_acc_classes:.3f}\t"
        # if len(metric_values) <= 10:
        #     report += f"acc@class{[round(a, 3) for a in metric_values.cpu().numpy()]}\t"

        return report


def check_nan(tensors):
    if torch.any(torch.isnan(tensors)):
        tensors = torch.nan_to_num(tensors)
    return tensors


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # https://discuss.pytorch.org/t/when-and-why-do-we-use-contiguous/47588
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

