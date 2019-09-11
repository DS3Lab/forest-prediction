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
from utils.util import save_images, NormalizeInverse
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

def update_acc(new_acc, acc_dict):
    if 0 <= new_acc < 0.1:
        acc_dict['acc0_10'] += 1
    elif 0.1 <= new_acc < 0.2:
        acc_dict['acc10_20'] += 1
    elif 0.2 <= new_acc < 0.3:
        acc_dict['acc20_30'] += 1
    elif 0.3 <= new_acc < 0.4:
        acc_dict['acc30_40'] += 1
    elif 0.4 <= new_acc < 0.5:
        acc_dict['acc40_50'] += 1
    elif 0.5 <= new_acc < 0.6:
        acc_dict['acc50_60'] += 1
    elif 0.6 <= new_acc < 0.7:
        acc_dict['acc60_70'] += 1
    elif 0.7 <= new_acc < 0.8:
        acc_dict['acc70_80'] += 1
    elif 0.8 <= new_acc < 0.9:
        acc_dict['acc80_90'] += 1
    else:
        acc_dict['acc90_100'] += 1

def update_individual_hists(data, target, hist, device, model):
    data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
    output = model(data)
    output_probs = F.sigmoid(output)
    binary_target = _threshold_outputs(target.data.cpu().numpy().flatten())
    output_binary = _threshold_outputs(
        output_probs.data.cpu().numpy().flatten())
    hist += _fast_hist(output_binary, binary_target)

def main(config):
    acc_dict = {
	'acc0_10': 0,
	'acc10_20': 0,
	'acc20_30': 0,
	'acc30_40': 0,
	'acc40_50': 0,
	'acc50_60': 0,
	'acc60_70': 0,
	'acc70_80': 0,
	'acc80_90': 0,
	'acc90_100': 0
    }
    logger = config.get_logger('test')
    # setup data_loader instances
    timelapse = config['data_loader_val']['args']['timelapse']
    input_type = config['data_loader_val']['args']['input_type']
    # img mode is used to do batch = 1 so it can minus the fc, and get the hansen loss
    img_mode = config['data_loader_val']['args']['img_mode']
    # if timelapse == 'quarter' or img_mode == 'cont':
    #     batch_size = 1
    # else:
    #     batch_size = 9
    batch_size = 1
    data_loader = getattr(module_data, config['data_loader_val']['type'])(
        input_dir=config['data_loader_val']['args']['input_dir'],
        label_dir=config['data_loader_val']['args']['label_dir'],
        batch_size=batch_size,
        years=config['data_loader_val']['args']['years'],
        qualities=config['data_loader_val']['args']['qualities'],
        timelapse=timelapse,
        max_dataset_size=float("inf"),
        shuffle=False,
        num_workers=1,
        training=False,
        testing=True,
        quarter_type="same_year",
        source='planet',
        input_type=input_type,
        img_mode=img_mode
    )

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
    pred_dir = os.path.join(pred_dir, 'normal')
    out_dir = os.path.join(pred_dir, timelapse)
    if not os.path.isdir(pred_dir):
        os.makedirs(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # out_dir = '/'.join(str(config.resume.absolute()).split('/')[:-1])
    # out_dir = os.path.join(out_dir, 'predictions')
    histq1 = np.zeros((2,2))
    histq2 = np.zeros((2,2))
    histq3 = np.zeros((2,2))
    histq4 = np.zeros((2,2))
    hist = np.zeros((2,2))
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
        # for i, (data, target) in enumerate(tqdm(data_loader)):
            init_time = time.time()
            loss = None
            imgs = batch['imgs']
            mask = batch['mask'] # 1, 1, 256, 256

            img0, img1, img2, img3 = imgs[0], imgs[1], imgs[2], imgs[3]
            uimg0, uimg1, uimg2, uimg3 = normalize_inverse(img0, (0.2311, 0.2838, 0.1752, (0.1265, 0.0955, 0.0891)), input_type), \
                    normalize_inverse(img1, (0.2311, 0.2838, 0.1752, (0.1265, 0.0955, 0.0891)), input_type), \
                    normalize_inverse(img2, (0.2311, 0.2838, 0.1752, (0.1265, 0.0955, 0.0891)), input_type), \
                    normalize_inverse(img3, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353)), input_type)
            data = torch.cat((img0, img1, img2, img3), 0)
            udata = torch.cat((uimg0, uimg1, uimg2, uimg3), 0)
            target = torch.cat((mask, mask, mask, mask), 0)
            # loss = batch['loss']
            # loss = torch.cat((loss, loss, loss, loss), 0)

            update_individual_hists(img0, mask, histq1, device, model)
            update_individual_hists(img1, mask, histq2, device, model)
            update_individual_hists(img2, mask, histq3, device, model)
            update_individual_hists(img3, mask, histq4, device, model)

            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)

            output_probs = F.sigmoid(output)
            binary_target = _threshold_outputs(target.data.cpu().numpy().flatten())
            output_binary = _threshold_outputs(
                output_probs.data.cpu().numpy().flatten())
            # print(output_binary.shape, 'SHAPEEE')
            mlz = _fast_hist(output_binary, binary_target)
            print('HELLO', mlz)
            hist += _fast_hist(output_binary, binary_target)
            images = {
                'img': udata.cpu().numpy(),
                'gt': target.cpu().numpy(),
                'pred': output_binary.reshape(-1, 1, 256, 256),
                'loss': loss.cpu().numpy() if loss is not None else np.zeros((target.shape))
            }

            print('prediction_time', time.time() - init_time)
            if timelapse == 'quarter' and input_type == 'two':
            # save sample images, or do something with output here
                print('Save images shape input', images['img'].shape)
                save_images(3, images, out_dir, i*batch_size, input_type)
            elif timelapse == 'quarter' and input_type == 'one':
                print('Save images shape input', images['img'].shape)
                save_images(4, images, out_dir, i*batch_size, input_type)
            elif timelapse == 'annual' and img_mode == 'cont':
                print('Save images shape input', images['img'].shape)
                save_images(2, images, out_dir, i*batch_size, input_type)
            else:
                print('Save images shape input', images['img'].shape)
                # save_images(3, images, out_dir, i*batch_size, input_type)
                save_images(1, images, out_dir, i*batch_size, input_type)
            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target.float()) * batch_size
            acc, acc_cls, mean_iu, fwavacc, precission, recall, f1_score = evaluate(hist=mlz)
            update_acc(acc, acc_dict)
    acc, acc_cls, mean_iu, fwavacc, precision, recall, f1_score = \
        evaluate(hist=hist)
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples,
        'acc': acc, 'mean_iu': mean_iu, 'fwavacc': fwavacc,
        'precision': precision, 'recall': recall, 'f1_score': f1_score
    }

    print(acc_dict)
    # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    logger.info(log)
    logger.info(acc_dict)

    accq1, acc_clsq1, mean_iuq1, fwavaccq1, precisionq1, recallq1, f1_scoreq1 = \
        evaluate(hist=histq1)
    accq2, acc_clsq2, mean_iuq2, fwavaccq2, precisionq2, recallq2, f1_scoreq2 = \
        evaluate(hist=histq2)
    accq3, acc_clsq3, mean_iuq3, fwavaccq3, precisionq3, recallq3, f1_scoreq3 = \
        evaluate(hist=histq3)
    accq4, acc_clsq4, mean_iuq4, fwavaccq4, precisionq4, recallq4, f1_scoreq4 = \
        evaluate(hist=histq4)
    logq1 = {'lossq1': total_loss / n_samples,
        'acc': accq1, 'mean_iu': mean_iuq1, 'fwavacc': fwavaccq1,
        'precision': precisionq1, 'recall': recallq1, 'f1_score': f1_scoreq1
    }
    logq2 = {'lossq2': total_loss / n_samples,
        'acc': accq2, 'mean_iu': mean_iuq2, 'fwavacc': fwavaccq2,
        'precision': precisionq2, 'recall': recallq2, 'f1_score': f1_scoreq2
    }
    logq3 = {'lossq3': total_loss / n_samples,
        'acc': accq3, 'mean_iu': mean_iuq3, 'fwavacc': fwavaccq3,
        'precision': precisionq3, 'recall': recallq3, 'f1_score': f1_scoreq3
    }
    logq4 = {'lossq4': total_loss / n_samples,
        'acc': accq4, 'mean_iu': mean_iuq4, 'fwavacc': fwavaccq4,
        'precision': precisionq4, 'recall': recallq4, 'f1_score': f1_scoreq4
    }
    logger.info(logq1)
    logger.info(logq2)
    logger.info(logq3)
    logger.info(logq4)


def normalize_inverse(batch, mean, std, input_type):

    with torch.no_grad():
        img = batch.clone()
        ubatch = torch.Tensor(batch.shape)
        if input_type == 'two':
            ubatch[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
            ubatch[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
            ubatch[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]
            ubatch[:, 3, :, :] = img[:, 3, :, :] * std[0] + mean[0]
            ubatch[:, 4, :, :] = img[:, 4, :, :] * std[1] + mean[1]
            ubatch[:, 5, :, :] = img[:, 5, :, :] * std[2] + mean[2]
        else: # one
            ubatch[:, 0, :, :] = img[:, 0, :, :] * std[0] + mean[0]
            ubatch[:, 1, :, :] = img[:, 1, :, :] * std[1] + mean[1]
            ubatch[:, 2, :, :] = img[:, 2, :, :] * std[2] + mean[2]
    return ubatch

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser(args)
    main(config)
