import os
import shutil
import time
from collections import OrderedDict

import torch

from utils.meter import AverageMeter, ImbalanceAccuracy, ProgressMeter, accuracy


class Trainer:
    def __init__(self, start_time, gpu_rank, device):
        self.start_time = start_time

        self.saved_values = dict()
        self.saved_values['best_eval_acc1'] = 0
        self.saved_values['best_eval_acc_details'] = None
        self.saved_values['best_train_acc1'] = 0
        self.saved_values['best_train_acc_details'] = None
        self.saved_values['train_errors'] = []
        self.saved_values['validate_errors'] = []
        self.saved_values['norm_head_med_tail'] = [[], [], []]

        self.start_epoch = 0
        self.pre_runtime = 0

        self.gpu_rank = gpu_rank
        self.device = device

    def load_checkpoint(self, learning_model, optimizer, file_path, device='cpu', logger=None, gpu_rank=0, ddp=False):
        if os.path.isfile(file_path):
            logger.info(f"=> loading checkpoint '{file_path}'", gpu_rank=gpu_rank)
            checkpoint = torch.load(file_path, map_location=device)
            learning_model.load_state_dict(_get_state_dict(checkpoint['state_dict_model'], ddp))
            optimizer.load_state_dict(checkpoint['optimizer'])

            self.start_epoch = checkpoint['epoch']
            self.saved_values = checkpoint['saved_values']
            self.pre_runtime = checkpoint['runtime']

            logger.info(f"=> loaded checkpoint '{file_path}' (trained {checkpoint['epoch']}+1 epochs)",
                        gpu_rank=gpu_rank)
            return checkpoint, learning_model, optimizer

        else:
            info = f"=> no checkpoint found at '{file_path}'"
            logger.info(info, gpu_rank=gpu_rank)
            raise FileNotFoundError(info)

    def train_model(self, learning_model, criterion, optimizer, dataset, lr_scheduler, num_epochs, logger, config):
        logger.info(f'====> training started\truntime: {time.time() - self.start_time + self.pre_runtime:.3f}',
                    gpu_rank=self.gpu_rank)
        epoch_time = time.time()
        for epoch in range(self.start_epoch, num_epochs):
            if dataset.dist_sampler is not None:
                dataset.dist_sampler.set_epoch(epoch)

            lr = lr_scheduler.update_lr(epoch)
            # training results will be different for models on different GPUs,
            # because different GPUs use different batches of data.
            self.train(learning_model, optimizer, criterion, dataset, epoch, lr, logger, config)
            runtime, is_best_v = self.validate(learning_model, criterion, dataset, logger, config)

            logger.info(f'Epoch_time: {time.time() - epoch_time:.3f}', gpu_rank=self.gpu_rank)
            # when running with multiple GPUs, all the models in different GPUs will be identical,
            # therefore, evaluation results are identical.
            is_last = (epoch + 1) == num_epochs
            # https://stackoverflow.com/questions/61037819/pytorch-how-to-save-and-load-model-from-distributeddataparallel-learning
            if config.output(self.gpu_rank):  # save for each node if running with multiple nodes
                if 'norm_head_med_tail' in self.saved_values:
                    if hasattr(learning_model, 'fc'):
                        # print(learning_model.fc.bias)
                        norm = torch.norm(learning_model.fc.weight, p=2, dim=1, keepdim=True)
                    else:  # the distributed training
                        norm = torch.norm(learning_model.module.fc.weight, p=2, dim=1, keepdim=True)
                    norm = norm.detach().cpu().numpy()
                    # logger.info(f"Classifier norm: {norm}")
                    self.saved_values['norm_head_med_tail'][0].append(norm[dataset.head_class_idx].mean())
                    if dataset.med_class_idx is not None:
                        self.saved_values['norm_head_med_tail'][1].append(norm[dataset.med_class_idx].mean())
                    self.saved_values['norm_head_med_tail'][2].append(norm[dataset.tail_class_idx].mean())

                self.save_checkpoint(epoch, learning_model, optimizer, runtime, is_best_v, is_last,
                                     config.dirs['model_dir'])
            epoch_time = time.time()
        return self.saved_values['train_errors'], self.saved_values['validate_errors'], self.saved_values[
            'norm_head_med_tail']

    def train(self, learning_model, optimizer, criterion, dataset, epoch, lr, logger, config):
        t_acc1, t_report = self._train(dataset, learning_model, criterion, optimizer, epoch, config)
        self.saved_values['train_errors'].append(100 - t_acc1.cpu().numpy())
        is_best_t = t_acc1 > self.saved_values['best_train_acc1']
        if is_best_t:
            self.saved_values['best_train_acc1'] = t_acc1
            self.saved_values['best_train_acc_details'] = t_report
        """training accuracy may be slightly different between different GPUs"""
        logger.info(
            f"* Train Acc@best: {self.saved_values['best_train_acc1']:.3f} "
            f"Error@best: {100 - self.saved_values['best_train_acc1']:0.3f}\t"
            f"{self.saved_values['best_train_acc_details']}"
            f"Runtime: {time.time() - self.start_time + self.pre_runtime:.3f}\tlr: {lr}", gpu_rank=0
        )

    def validate(self, learning_model, criterion, dataset, logger, config, evaluate=False):
        is_best_v = None
        v_acc1, v_report = self._validate(dataset, learning_model, criterion, config)
        if evaluate:
            assert abs(
                v_acc1 - self.saved_values['best_eval_acc1']) < 1e-8, f"{v_acc1 - self.saved_values['best_eval_acc1']}"
        else:
            self.saved_values['validate_errors'].append(100 - v_acc1.cpu().numpy())
            is_best_v = v_acc1 > self.saved_values['best_eval_acc1']
            if is_best_v:
                self.saved_values['best_eval_acc1'] = v_acc1
                self.saved_values['best_eval_acc_details'] = v_report
        runtime = time.time() - self.start_time + self.pre_runtime
        logger.info(
            f"* Eval Acc@best: {self.saved_values['best_eval_acc1']:.3f} "
            f"Error@best: {100 - self.saved_values['best_eval_acc1']:0.3f}\t"
            f"{self.saved_values['best_eval_acc_details']}"
            f"Runtime: {runtime:.3f}\t", gpu_rank=self.gpu_rank
        )
        return runtime, is_best_v

    def _train(self, dataset, learning_model, criterion, optimizer, epoch, config) -> [float, str]:
        batch_time = AverageMeter('Batch_time', ':6.3f')
        data_time = AverageMeter('Data_time', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')
        progress = ProgressMeter(
            len(dataset.train_dataloader),
            [batch_time, data_time, losses, top1, top5],
            prefix=f"Train epoch: [{epoch}]")
        imb_process = ImbalanceAccuracy(dataset, self.device)
        num_classes = dataset.num_classes

        # switch to train mode
        learning_model.train()

        # training_data_num = len(train_loader.dataset)
        # end_steps = int(training_data_num / train_loader.batch_size)

        end = time.time()
        # print(f"Start running training..{device}")
        for i, (images, targets) in enumerate(dataset.train_dataloader):
            # measure data loading time
            data_time.update(time.time() - end)
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            loss, output = criterion(learning_model(images), targets)
            imb_process.update(targets, output)
            acc1, acc5 = accuracy(output, targets, topk=(1, min(num_classes, 5)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            # probs = torch.softmax(output, dim=1)
            # y = torch.nn.functional.one_hot(targets, dataset.num_classes)
            # bias_grads = (probs - y).sum(dim=0) / len(targets)
            # print(f"gradients: {bias_grads}")
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.config['trainer']['print_freq'] == 0:
                progress.display(i, config.logger, self.gpu_rank)

        report = imb_process.calculate(config.logger, self.gpu_rank)
        top1_accuracy = top1.avg
        # manually delete the variables for saving memory.
        del batch_time, data_time, losses, top1, top5, progress

        return top1_accuracy, report

    def _validate(self, dataset, learning_model, criterion, config) -> [float, str]:
        batch_time = AverageMeter('Batch_time', ':6.3f')
        data_time = AverageMeter('Data_time', ':6.3f')
        losses = AverageMeter('Loss', ':.3f')
        top1 = AverageMeter('Acc@1', ':6.3f')
        top5 = AverageMeter('Acc@5', ':6.3f')
        progress = ProgressMeter(
            len(dataset.eval_dataloader),
            [batch_time, data_time, losses, top1, top5],
            prefix='Eval: ')
        imb_process = ImbalanceAccuracy(dataset, self.device)
        # calibration = Calibration()
        # switch to evaluate mode
        learning_model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(dataset.eval_dataloader):
                data_time.update(time.time() - end)
                target = target.to(self.device, non_blocking=True)
                images = images.to(self.device, non_blocking=True)

                loss, output = criterion(learning_model(images), target)

                # measure accuracy and record loss
                losses.update(loss.item(), images.size(0))
                acc1, acc5 = accuracy(output, target, topk=(1, min(dataset.num_classes, 5)))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))

                imb_process.update(target, output)

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % config.config['trainer']['print_freq'] == 0:
                    progress.display(i, config.logger, self.gpu_rank)
            report = imb_process.calculate(config.logger, self.gpu_rank)
            top1_accuracy = top1.avg

            del batch_time, data_time, losses, top1, top5, progress

        return top1_accuracy, report

    def save_checkpoint(self, epoch, learning_model, optimizer, runtime, is_best_v, is_last, model_dir):
        checkpoint = {
            'epoch': epoch,
            'state_dict_model': learning_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'runtime': runtime,
            'saved_values': self.saved_values
        }

        filename = model_dir / 'current.pth.tar'
        torch.save(checkpoint, filename)
        if is_best_v:
            shutil.copyfile(filename, model_dir / 'model_best_eval.pth.tar')
        if is_last:
            if is_last:
                shutil.copyfile(filename, model_dir / 'last.pth.tar')


def _get_state_dict(state_dict, ddp):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if ddp:
            if 'module.' in k:
                return state_dict
            else:
                name = 'module.' + k  # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
        else:
            if 'module.' in k:
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            else:
                return state_dict
    state_dict = new_state_dict
    return state_dict
