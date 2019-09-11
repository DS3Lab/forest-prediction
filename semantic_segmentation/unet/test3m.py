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
from data_loader import utils3m

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

def update_individual_hists(data, target, hist, device, model):
    data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
    output = model(data)
    output_probs = F.sigmoid(output)
    binary_target = _threshold_outputs(target.data.cpu().numpy().flatten())
    output_binary = _threshold_outputs(
        output_probs.data.cpu().numpy().flatten())
    hist += _fast_hist(output_binary, binary_target)
    return output_binary.reshape(-1, 1, 256, 256)

def main(config):
    logger = config.get_logger('test')
    # setup data_loader instances
    batch_size = 1
    timelapse = 'annual'
    input_type = 'same'
    img_mode = 'single'
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
    pred_dir = os.path.join(pred_dir, '3m')
    out_dir = os.path.join(pred_dir, timelapse)
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    histy = np.zeros((2,2))
    histq1 = np.zeros((2,2))
    histq2 = np.zeros((2,2))
    histq3 = np.zeros((2,2))
    histq4 = np.zeros((2,2))

    mean, std = (0.2311, 0.2838, 0.1752), (0.1265, 0.0955, 0.0891)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
        # for i, (data, target) in enumerate(tqdm(data_loader)):
            init_time = time.time()
            loss = None
            original_mask = batch['mask']
            big_mask = batch['big_mask']
            tile_coord = batch['key'].cpu().numpy()
            tile_x, tile_y = tile_coord[0], tile_coord[1]
            beg_x, beg_y, num_tiles = utils3m.zoom2zoom(12, tile_x, tile_y, 16)

            keys = batch.keys()
            recreate_mask = {}
            recreate_mask_pred = {}
            for key in keys:
                if key in ['mask_arr', 'big_mask', 'key'] :
                    continue
                imgs = batch[key]
                # q1, q2, q3, q4, annual, mask = imgs['q1'], imgs['q2'], \
                #     imgs['q3'], imgs['q4'], imgs['annual'], imgs['mask']
                # uq1, uq2, uq3, uq4, uannual = normalize_inverse(q1, (mean, std)), \
                #     normalize_inverse(q2, (mean, std)), \
                #     normalize_inverse(q3, (mean, std)), \
                #     normalize_inverse(q4, (mean, std)), \
                #     normalize_inverse(annual, (mean, std))
                annual, mask = imgs['annual'], imgs['mask']
                # pred_q1 = update_individual_hists(q1, mask, histq1, device, model)
                # pred_q2 = update_individual_hists(q2, mask, histq1, device, model)
                # pred_q3 = update_individual_hists(q3, mask, histq1, device, model)
                # pred_q4 = update_individual_hists(q4, mask, histq1, device, model)
                pred_annual = update_individual_hists(annual, mask, histy, device, model)
                recreate_mask[key] = np.squeeze(mask.cpu().numpy(), 0)
                recreate_mask_pred[key] = np.squeeze(pred_annual.cpu().numpy(), 0)
                print(recreate_mask[key].shape, recreate_mask_pred[key].shape, 'DEBUGGING RECREATION SHAPE!!!!!')

                mask_recreation_gt = utils3m.reconstruct_tile(recreate_mask)
                mask_recreation_pred = utils3m.reconstruct_tile(recreate_mask_pred)

                images = {
                    'mask_recreation_gt': mask_recreation_gt,
                    'mask_recreation_pred': mask_recreation_pred,
                    'mask': mask.cpu().numpy()
                }
                save_images3m(2, images, out_dir, i*batch_size)
            break
            # computing loss, metrics on test set
            # loss = loss_fn(output, target)
            # batch_size = data.shape[0]
            # total_loss += loss.item() * batch_size
            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(output, target.float()) * batch_size

    # acc, acc_cls, mean_iu, fwavacc, precision, recall, f1_score = \
    #     evaluate(hist=hist)
    n_samples = len(data_loader.sampler)

    accq1, acc_clsq1, mean_iuq1, fwavaccq1, precisionq1, recallq1, f1_scoreq1 = \
        evaluate(hist=histq1)
    accq2, acc_clsq2, mean_iuq2, fwavaccq2, precisionq2, recallq2, f1_scoreq2 = \
        evaluate(hist=histq2)
    accq3, acc_clsq3, mean_iuq3, fwavaccq3, precisionq3, recallq3, f1_scoreq3 = \
        evaluate(hist=histq3)
    accq4, acc_clsq4, mean_iuq4, fwavaccq4, precisionq4, recallq4, f1_scoreq4 = \
        evaluate(hist=histq4)

    accy, acc_clsy, mean_iuy, fwavaccy, precisiony, recally, f1_scorey = \
        evaluate(hist=histy)

    logy = {'lossy': 'total_loss / n_samples',
        'acc': accy, 'mean_iu': mean_iuy, 'fwavacc': fwavaccy,
        'precision': precisiony, 'recall': recally, 'f1_score': f1_scorey
    }

    logq1 = {'lossq1': 'total_loss / n_samples',
        'acc': accq1, 'mean_iu': mean_iuq1, 'fwavacc': fwavaccq1,
        'precision': precisionq1, 'recall': recallq1, 'f1_score': f1_scoreq1
    }
    logq2 = {'lossq2': 'total_loss / n_samples',
        'acc': accq2, 'mean_iu': mean_iuq2, 'fwavacc': fwavaccq2,
        'precision': precisionq2, 'recall': recallq2, 'f1_score': f1_scoreq2
    }
    logq3 = {'lossq3': 'total_loss / n_samples',
        'acc': accq3, 'mean_iu': mean_iuq3, 'fwavacc': fwavaccq3,
        'precision': precisionq3, 'recall': recallq3, 'f1_score': f1_scoreq3
    }
    logq4 = {'lossq4': 'total_loss / n_samples',
        'acc': accq4, 'mean_iu': mean_iuq4, 'fwavacc': fwavaccq4,
        'precision': precisionq4, 'recall': recallq4, 'f1_score': f1_scoreq4
    }
    logger.info(logq1)
    logger.info(logq2)
    logger.info(logq3)
    logger.info(logq4)
    logger.info(logy)


    # log = {'loss': total_loss / n_samples,
    #     'acc': acc, 'mean_iu': mean_iu, 'fwavacc': fwavacc,
    #     'precision': precision, 'recall': recall, 'f1_score': f1_score
    # }
    #
    # print(acc_dict)
    # # log.update({
    #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    # })
    # logger.info(log)
    # logger.info(acc_dict)



def normalize_inverse(batch, mean, std):
    with torch.no_grad():
        img = batch.clone()
        ubatch = torch.Tensor(batch.shape)
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
