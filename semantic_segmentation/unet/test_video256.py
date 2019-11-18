"""
Main entry point for the binary segmentation task for video prediction results.
It takes following arguments:
-r full path to the trained model
-d specifies the GPU ids to be used (it takes max n_gpus defined in config.json)

Note: if the model was trained on n GPU, the testing is expecting n GPU.
Note2: in the path_of_saved_model directory, it expects a config.json
    It is saved by default in the training

Example usage:
```
python test.py -r path_of_saved_model/model.pth -d [gpu_id,]
```
"""
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
from trainer import evaluate, fast_hist, threshold_outputs
from utils.util import save_simple_images, save_double_images
from torch.nn import functional as F

def update_individual_hists(data, target, hist, device, model):
    data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
    output = model(data)
    output_probs = F.sigmoid(output)
    binary_target = threshold_outputs(target.data.cpu().numpy().flatten())
    output_binary = threshold_outputs(
        output_probs.data.cpu().numpy().flatten())
    hist += fast_hist(output_binary, binary_target)
    return output_binary.reshape(-1, 1, 256, 256)

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
        video_dir=config['data_loader_val']['args']['video_dir'],
        batch_size=batch_size,
        max_dataset_size=max_dataset_size,
        shuffle=False,
        num_workers=1
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
    # out_dir = os.path.join(pred_dir, 'video_loss_last_three')
    out_dir = os.path.join(pred_dir, 'rm')
    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    # out_dir = '/'.join(str(config.resume.absolute()).split('/')[:-1])
    # out_dir = os.path.join(out_dir, 'predictions')
    hist = np.zeros((2,2))
    hist2013 = np.zeros((2,2))
    hist2014 = np.zeros((2,2))
    hist2015 = np.zeros((2,2))
    hist2016 = np.zeros((2,2))
    hist2017 = np.zeros((2,2))

    # This script only supports batch=1
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            if i not in [0, 84, 55]:
                continue
            # if i not in [18, 43, 51, 61, 73, 84, 85, 88, 116, 124, 198, 201, 214, 245, 325, 330]:
            # if i not in [84, 85, 88, 116, 124, 198, 201, 214, 245, 325]:
            #     continue
        # for i, (data, target) in enumerate(tqdm(data_loader)):
            init_time = time.time()
            loss = None
            img_arr2013, mask_arr2013 = batch['2013']['img_arr'], batch['2013']['mask_arr']
            img_arr2014, mask_arr2014 = batch['2014']['img_arr'], batch['2014']['mask_arr']
            img_arr2015, mask_arr2015 = batch['2015']['img_arr'], batch['2015']['mask_arr']
            img_arr2016, mask_arr2016 = batch['2016']['img_arr'], batch['2016']['mask_arr']
            img_arr2017, mask_arr2017 = batch['2017']['img_arr'], batch['2017']['mask_arr']

            img_arr2015p, _ = batch['2015p']['img_arr'], batch['2015']['mask_arr']
            img_arr2016p, _ = batch['2016p']['img_arr'], batch['2016']['mask_arr']
            img_arr2017p, _ = batch['2017p']['img_arr'], batch['2017']['mask_arr']

            uimg_arr2013, uimg_arr2014, uimg_arr2015, uimg_arr2016, uimg_arr2017 = \
                normalize_inverse(img_arr2013, landsat_mean, landsat_std), \
                normalize_inverse(img_arr2014, landsat_mean, landsat_std), \
                normalize_inverse(img_arr2015, landsat_mean, landsat_std), \
                normalize_inverse(img_arr2016, landsat_mean, landsat_std), \
                normalize_inverse(img_arr2017, landsat_mean, landsat_std)
            uimg_arr2015p, uimg_arr2016p, uimg_arr2017p = normalize_inverse(img_arr2015p, landsat_mean, landsat_std), \
                normalize_inverse(img_arr2016p, landsat_mean, landsat_std), \
                normalize_inverse(img_arr2017p, landsat_mean, landsat_std), \

            pred2013 = update_individual_hists(img_arr2013, mask_arr2013, hist2013, device, model)
            pred2014 = update_individual_hists(img_arr2014, mask_arr2014, hist2014, device, model)
            pred2015 = update_individual_hists(img_arr2015, mask_arr2015, hist2015, device, model)
            pred2016 = update_individual_hists(img_arr2016, mask_arr2016, hist2016, device, model)
            pred2017 = update_individual_hists(img_arr2017, mask_arr2017, hist2017, device, model)

            pred2015p = update_individual_hists(img_arr2015p, mask_arr2015, hist2015, device, model)
            pred2016p = update_individual_hists(img_arr2016p, mask_arr2016, hist2016, device, model)
            pred2017p = update_individual_hists(img_arr2017p, mask_arr2017, hist2017, device, model)

            images = {
                '2013':{
                    'img': uimg_arr2013.cpu().numpy(),
                    'gt': mask_arr2013.cpu().numpy(),
                    'pred': pred2013
                },
                '2014':{
                    'img': uimg_arr2014.cpu().numpy(),
                    'gt': mask_arr2014.cpu().numpy(),
                    'pred': pred2014
                },
                '2015':{
                    'img': uimg_arr2015.cpu().numpy(),
                    'gt': mask_arr2015.cpu().numpy(),
                    'pred': pred2015
                },
                '2016':{
                    'img': uimg_arr2016.cpu().numpy(),
                    'gt': mask_arr2016.cpu().numpy(),
                    'pred': pred2016
                },
                '2017':{
                    'img': uimg_arr2017.cpu().numpy(),
                    'gt': mask_arr2017.cpu().numpy(),
                    'pred': pred2017
                },
                '2015p':{
                    'img': uimg_arr2015p.cpu().numpy(),
                    'gt': mask_arr2015.cpu().numpy(),
                    'pred': pred2015p
                },
                '2016p':{
                    'img': uimg_arr2016p.cpu().numpy(),
                    'gt': mask_arr2016.cpu().numpy(),
                    'pred': pred2016p
                },
                '2017p':{
                    'img': uimg_arr2017p.cpu().numpy(),
                    'gt': mask_arr2017.cpu().numpy(),
                    'pred': pred2017p
                },
            }
            save_video_images256(images, out_dir, i*batch_size)

    acc2013, acc_cls2013, mean_iu2013, fwavacc2013, precision2013, recall2013, f1_score2013 = \
        evaluate(hist=hist2013)

    acc2014, acc_cls2014, mean_iu2014, fwavacc2014, precision2014, recall2014, f1_score2014 = \
        evaluate(hist=hist2014)

    acc2015, acc_cls2015, mean_iu2015, fwavacc2015, precision2015, recall2015, f1_score2015 = \
        evaluate(hist=hist2015)

    acc2016, acc_cls2016, mean_iu2016, fwavacc2016, precision2016, recall2016, f1_score2016 = \
        evaluate(hist=hist2016)
    acc2017, acc_cls2017, mean_iu2017, fwavacc2017, precision2017, recall2017, f1_score2017 = \
        evaluate(hist=hist2017)
    n_samples = len(data_loader.sampler)

    log2013 = {'loss2013': -1,
        'acc': acc2013, 'mean_iu': mean_iu2013, 'fwavacc': fwavacc2013,
        'precision': precision2013, 'recall': recall2013, 'f1_score': f1_score2013
    }

    log2014 = {'loss2014': -1,
        'acc': acc2014, 'mean_iu': mean_iu2014, 'fwavacc': fwavacc2014,
        'precision': precision2014, 'recall': recall2014, 'f1_score': f1_score2014
    }

    log2015 = {'loss2015': -1,
        'acc': acc2015, 'mean_iu': mean_iu2015, 'fwavacc': fwavacc2015,
        'precision': precision2015, 'recall': recall2015, 'f1_score': f1_score2015
    }

    log2016 = {'loss2016': -1,
        'acc': acc2016, 'mean_iu': mean_iu2016, 'fwavacc': fwavacc2016,
        'precision': precision2016, 'recall': recall2016, 'f1_score': f1_score2016
    }

    log2017 = {'loss2017': -1,
        'acc': acc2017, 'mean_iu': mean_iu2017, 'fwavacc': fwavacc2017,
        'precision': precision2017, 'recall': recall2017, 'f1_score': f1_score2017
    }

    logger.info(log2013)
    logger.info(log2014)
    logger.info(log2015)
    logger.info(log2016)
    logger.info(log2017)

def normalize_inverse(batch, mean, std, input_type='one'):

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
