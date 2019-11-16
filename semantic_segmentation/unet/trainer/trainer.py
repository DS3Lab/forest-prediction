"""
Main training engine used to train the models. 
"""
import time
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, make_grid_2


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        """
    def __init__(self, model, loss, metrics, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metrics, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.save_train_img_step = self.log_step * 4
        
        # Binarize NN output
        self.output_threshold = 0.3

    def _eval_metrics(self, output, target):
        """
        Make an evaluation of all the specified metrics and store it in an ndarray.
        :params
            output: prediction of the model
            target: ground truth
        """
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            m = metric(output, target.float())
            acc_metrics[i] += metric(output, target.float())
            self.writer.add_scalar('{}'.format(metric.__name__), acc_metrics[i])
        return acc_metrics

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log

            The metrics in log must have the key 'metrics'.
        """
        self.model.train()
        total_loss = 0
        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.writer.add_scalar('loss', loss.item())
            total_loss += loss.item()
            total_metrics += self._eval_metrics(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
            
            if batch_idx == self.len_epoch:
                break

        log = {
            'loss': total_loss / self.len_epoch,
            'metrics': (total_metrics / self.len_epoch).tolist()
        }

        val_log = {}
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(val_log)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log, val_log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        total_val_loss = 0
        hist = np.zeros((2,2))

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data) # logits
                output_probs = F.sigmoid(output)
                loss = self.loss(output, target)
                target = self._threshold_outputs(target.data.cpu().numpy().flatten())
                output_binary = self._threshold_outputs(
                    output_probs.data.cpu().numpy().flatten())

                # update the confusion matrix
                hist += _fast_hist(output_binary, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.writer.add_scalar('loss', loss.item())
                total_val_loss += loss.item()
            acc, acc_cls, mean_iu, fwavacc, precision, recall, f1_score = \
                evaluate(hist=hist)

        # add histogram of model parameters to tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        return {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'acc': acc, 'mean_iu': mean_iu, 'fwavacc': fwavacc,
            'precision': precision, 'recall': recall, 'f1_score': f1_score
        }

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _threshold_outputs(self, outputs):
        idx = outputs > self.output_threshold
        outputs = np.zeros(outputs.shape, dtype=np.int8)
        outputs[idx] = 1
        return outputs

# TODO: This should be in the loss script. Just here to fast testing

def _fast_hist(outputs, targets, num_classes=2):
    mask = (targets >= 0) & (targets < num_classes)
    hist = np.bincount(
        num_classes * targets[mask].astype(int) +
        outputs[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist

def evaluate(outputs=None, targets=None, hist=None, num_classes=2):
    if hist is None:
        hist = np.zeros((num_classes, num_classes))
        for lp, lt in zip(outputs, targets):
            hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    eps = 1e-10
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    true_pos = hist[1, 1]
    false_pos = hist[0, 1]
    false_neg = hist[1, 0]
    precision = true_pos / (true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)
    f1_score = 2. * ((precision * recall) / (precision + recall + eps))

    return acc, acc_cls, mean_iu, fwavacc, precision, recall, f1_score
