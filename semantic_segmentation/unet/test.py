"""
Main entry point for the binary segmentation task.
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

def get_output_dir(img_dir):
    """
    Set output dir according to the image test directory
    """
    if 'pix2pix' in img_dir:
        return 'pix2pix'
    elif 'landsat' in img_dir:
        return 'landsat'
    else:
        return 'planet'

def create_loss(fc0, fc1):
    """
    Create forest loss from forest cover at time 0 and 1
    Params:
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
        batch_size=batch_size,
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
    out_dir = os.path.join(pred_dir, get_output_dir(config['data_loader_val']['args']['img_dir']))

    if not os.path.isdir(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    hist = np.zeros((2,2))
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
        # for i, (data, target) in enumerate(tqdm(data_loader)):
            init_time = time.time()
            loss = None
            data, target = batch
            udata = normalize_inverse(data, landsat_mean, landsat_std)

            data, target = data.to(device, dtype=torch.float), target.to(device, dtype=torch.float)
            output = model(data)

            output_probs = F.sigmoid(output)
            binary_target = threshold_outputs(target.data.cpu().numpy().flatten())
            output_binary = threshold_outputs(
                output_probs.data.cpu().numpy().flatten())

            np.save('planet_prediction_{}.npy'.format(i), output_binary.reshape(-1,1,256,256)[0,0,:,:])
            # print(output_binary.shape, 'SHAPEEE')
            hist += fast_hist(output_binary, binary_target)
            images = {
                'img': udata.cpu().numpy(),
                'gt': target.cpu().numpy(),
                'pred': output_binary.reshape(-1, 1, 256, 256),
            }

            # Save single images (C=3) or double images (C=6)
            if images['img'].shape[1] == 3:
                save_simple_images(3, images, out_dir, i*batch_size)
            else:
                save_double_images(3, images, out_dir, i*batch_size)
            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size

    # Update binary segmentation metrics
    acc, acc_cls, mean_iu, fwavacc, precision, recall, f1_score = \
        evaluate(hist=hist)
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples,
        'acc': acc, 'mean_iu': mean_iu, 'fwavacc': fwavacc,
        'precision': precision, 'recall': recall, 'f1_score': f1_score
    }
    logger.info(log)

def normalize_inverse(batch, mean, std):
    """
    Performs the inverse of normalization. Returns unnormalized batch.
        :param batch: batch from DataLoader, NCHW format
        :param mean: tensor of shape (3,)
        :param std: tensor of shape (3,)
    """
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
