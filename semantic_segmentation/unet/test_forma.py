import argparse
import torch
import os
import numpy as np
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import time
from parse_config import ConfigParser
from utils.util import NormalizeInverse, save_result_images
from torch.nn import functional as F


def _threshold_outputs(outputs, output_threshold=0.3):
    idx = outputs > output_threshold
    outputs = np.zeros(outputs.shape, dtype=np.int8)
    outputs[idx] = 1
    return outputs

def _fast_hist(outputs, targets, num_classes=2):
    print(outputs.shape, targets.shape)
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

def get_output_dir(img_dir):
    return 'forma'

def create_loss(fc0, fc1):
    """
    fc0: forest cover at time year t-1
    fc1: forest cover at time year t
    fc0 - fc1 = forest loss at year t
    """
    fl0 = fc0 - fc1
    gain_mask = np.where(fl0 < 0) # there is forest in t+1 but not in t
    fl0[gain_mask] = 0
    return fl0

def main(config):
    logger = config.get_logger('test')
    # setup data_loader instances
    batch_size = 1
    if config['data_loader_val']['args']['max_dataset_size'] == 'inf':
        max_dataset_size = float('inf')
    else:
        max_dataset_size = config['data_loader_val']['args']['max_dataset_size']
    data_loader = getattr(module_data, config['data_loader_val']['type'])(
        img_dir=config['data_loader_val']['args']['img_dir'],
        label_dir=config['data_loader_val']['args']['label_dir'],
        batch_size=1,
        years=config['data_loader_val']['args']['years'],
        max_dataset_size=max_dataset_size,
        shuffle=False,
        num_workers=1,
    )
    landsat_mean, landsat_std = (0.3326, 0.3570, 0.2224), (0.1059, 0.1086, 0.1283)
    # build model architecture
    model = config.initialize('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = config.initialize('loss', module_loss)
    # loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    pred_dir = '/'.join(str(config.resume.absolute()).split('/')[:-1])
    # pred_dir = os.path.join(pred_dir, 'predictions')
    out_dir = os.path.join(pred_dir, get_output_dir(config['data_loader_val']['args']['img_dir']))
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # out_dir = '/'.join(str(config.resume.absolute()).split('/')[:-1])
    # out_dir = os.path.join(out_dir, 'predictions')
    hist = np.zeros((2,2))
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
        # for i, (data, target) in enumerate(tqdm(data_loader)):
            init_time = time.time()
            loss = None
            img_arr0, img_arr1, forma, hansen = batch
            uimg_arr0, = normalize_inverse(img_arr0, landsat_mean, landsat_std)
            uimg_arr1, = normalize_inverse(img_arr1, landsat_mean, landsat_std)
            forma, hansen = forma.to(device, dtype=torch.float), hansen.to(device, dtype=torch.float)

            binary_forma = _threshold_outputs(forma.data.cpu().numpy().flatten())
            binary_hansen = _threshold_outputs(hansen.data.cpu().numpy().flatten())

            # print(output_binary.shape, 'SHAPEEE')
            mlz = _fast_hist(binary_forma, binary_hansen)
            hist += mlz
            images = {
                'img0': uimg_arr0.cpu().numpy(),
                'img1': uimg_arr1.cpu().numpy(),
                'forma': forma.cpu().numpy(),
                'hansen': hansen.cpu().numpy()
            }
            save_forma_images(1, images, out_dir, i*batch_size)

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target.float()) * batch_size

    acc, acc_cls, mean_iu, fwavacc, precision, recall, f1_score = \
        evaluate(hist=hist)
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples,
        'acc': acc, 'mean_iu': mean_iu, 'fwavacc': fwavacc,
        'precision': precision, 'recall': recall, 'f1_score': f1_score
    }

    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    logger.info(log)

def normalize_inverse(batch, mean, std):
    with torch.no_grad():
        img = batch.clone()
        ubatch = torch.Tensor(batch.shape)
        if img.shape[1] == 3: # 1 image
            ubatch[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
            ubatch[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
            ubatch[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]
        else: # 2 input images
            ubatch[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
            ubatch[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
            ubatch[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]
            ubatch[:, 3, :, :] = img[:, 3, :, :] * std[0] + mean[0]
            ubatch[:, 4, :, :] = img[:, 4, :, :] * std[1] + mean[1]
            ubatch[:, 5, :, :] = img[:, 5, :, :] * std[2] + mean[2]
    return ubatch

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser(args)
    main(config)
