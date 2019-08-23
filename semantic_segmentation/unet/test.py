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


def init_data_loader(source, config):
    # TODO: FINISH
    if source == 'planet':
        data_loader = getattr(module_data, config['data_loader_val']['type'])(
        data_dir=config['data_loader_val']['args']['data_dir'],
        batch_size=batch_size,
        years=config['data_loader_val']['args']['years'],
        qualities=config['data_loader_val']['args']['qualities'],
        timelapse=timelapse,
        # max_dataset_size=config['data_loader_val']['args']['max_dataset_size'],
        # max_dataset_size=9,
        max_dataset_size=900,
        shuffle=False,
        num_workers=32,
        training=False,
        testing=True,
        quarter_type="same_year"
    )
    else: # landsat
        data_loader = getattr(module_data,
                config['data_loader_val']['type'])(
                        data_dir=config['data_loader_val']['args']['data_dir'],
                        batch_size=batch_size,
                        max_dataset_size='inf',
                        shuffle=False)


def main(config):
    logger = config.get_logger('test')
    # setup data_loader instances
    timelapse = config['data_loader_val']['args']['timelapse']
    input_type = config['data_loader_val']['args']['input_type']
    img_mode = config['data_loader_val']['args']['img_mode']
    if timelapse == 'quarter' or img_mode == 'cont':
        batch_size = 1
    else:
        batch_size = 9
    data_loader = getattr(module_data, config['data_loader_val']['type'])(
        input_dir=config['data_loader_val']['args']['input_dir'],
        label_dir=config['data_loader_val']['args']['label_dir'],
        batch_size=batch_size,
        years=config['data_loader_val']['args']['years'],
        qualities=config['data_loader_val']['args']['qualities'],
        timelapse=timelapse,
        max_dataset_size=900,
        shuffle=False,
        num_workers=32,
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
    pred_dir = os.path.join(pred_dir, 'landsat2')
    out_dir = os.path.join(pred_dir, timelapse)
    if not os.path.isdir(pred_dir):
        os.makedirs(out_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # out_dir = '/'.join(str(config.resume.absolute()).split('/')[:-1])
    # out_dir = os.path.join(out_dir, 'predictions')
    hist = np.zeros((2,2))
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
        # for i, (data, target) in enumerate(tqdm(data_loader)):
            init_time = time.time()
            if timelapse == 'quarter' and input_type == 'two':
                imgs = batch['imgs']
                # mask = batch['mask'].unsqueeze(0) # 1, 1, 256, 256
                mask = batch['mask'] # 1, 1, 256, 256

                img0, img1, img2 = imgs[0], imgs[1], imgs[2]
                uimg0, uimg1, uimg2 = normalize_inverse(img0, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353), input_type), \
                        normalize_inverse(img1, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353), input_type), \
                        normalize_inverse(img2, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353), input_type)
                data = torch.cat((img0, img1, img2), 0)
                udata = torch.cat((uimg0, uimg1, uimg2), 0)

                target = torch.cat((mask, mask, mask), 0)
                print('MINI BATCH', data.shape)
            elif timelapse == 'quarter' and input_type == 'one':
                imgs = batch['imgs']
                # mask = batch['mask'].unsqueeze(0) # 1, 1, 256, 256
                mask = batch['mask'] # 1, 1, 256, 256

                img0, img1, img2, img3 = imgs[0], imgs[1], imgs[2], imgs[3]
                uimg0, uimg1, uimg2, uimg3 = normalize_inverse(img0, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353), input_type), \
                        normalize_inverse(img1, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353), input_type), \
                        normalize_inverse(img2, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353), input_type), \
                        normalize_inverse(img3, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353), input_type)
                data = torch.cat((img0, img1, img2, img3), 0)
                udata = torch.cat((uimg0, uimg1, uimg2, uimg3), 0)

                target = torch.cat((mask, mask, mask, mask), 0)
                print('MINI BATCH', data.shape)
            else: # 'annual'
                if img_mode == 'same':
                    data, target = batch
                    print('MINI BATCH', data.shape, target.shape)
                    udata = normalize_inverse(data, (0.2311, 0.2838, 0.1752),
                        (0.1265, 0.0955, 0.0891), input_type)
                else:
                    imgs = batch['imgs']
                    # mask = batch['mask'].unsqueeze(0) # 1, 1, 256, 256
                    mask = batch['mask'] # 1, 1, 256, 256
                    loss = batch['loss']
                    img0, img1 = imgs[0], imgs[1]
                    uimg0, uimg1 = normalize_inverse(img0, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353), input_type), \
                            normalize_inverse(img1, (0.2397, 0.2852, 0.1837), (0.1685, 0.1414, 0.1353), input_type)

                    data = torch.cat((img0, img1), 0)
                    udata = torch.cat((uimg0, uimg1), 0)

                    target = torch.cat((mask[0], mask[1]), 0)
                    loss = torch.cat((loss, loss), 0)
                    print('MINI BATCH', data.shape)

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
                'loss': loss.cpu().numpy() if loss else np.zeros((gt.shape))
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
                save_images(3, images, out_dir, i*batch_size, input_type)

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
