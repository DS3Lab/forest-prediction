"""
Creates a class that processes config.json

A config json consists of:

- name: directory where experiments are saved
- n_gpu: number of GPUs used
- arch:
    - type: name of model in unet/model/model.py
    - args: argument that the model takes
- data_loader_train:
    - type: name of the data loader in unet/data_loader/data_loaders.py
    - args: arguments of data loader
- data_loader_val:
    - type: name of the data loader in unet/data_loader/data_loaders.py
    - args: arguments of data loader
- optimizer:
    - type: name of the optimizer from torch.optim package
    - args: arguments of optimizer
- loss:
    - type: name of the loss in unet/model/loss.py
    - args: arguments of the loss
- metrics: [ <name of the method in unet/model/metric.py>, ... ]
- lr_scheduler: 
    - type: name of the scheduler from pytorch from torch.optim package
    - args: arguments of the scheduler
- trainer: 
    - epochs: number of epochs to train
    - save_dir: path of the root directory to save experiments
    - save_period: checkpoint is saved after each `save_period` epoch
    - verbosity: console logging stages (0,1,2)
    - keep_last: number of most recent models to keep
    - monitor: takes `<mode><space><metric>` and saves the best model on validation dataset with equation: mode(metric)
        `e.g. min val_loss` saves the model that has minimal validation loss as model_best.pth
    - early_stop: number of epochs to check - if validation didn't improved it stops
    - tensorboard: boolean - whether to use tensorboard
"""


import os
import logging
from pathlib import Path
from functools import reduce
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json

"""
ConfigParser is used in train.py and test.py and takes in config.json files with `-c`
"""
class ConfigParser:
    def __init__(self, args, options='', timestamp=True):
        # parse default and custom cli options
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume:
            self.resume = Path(args.resume)
            self.cfg_fname = self.resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            self.resume = None
            self.cfg_fname = Path(args.config)

        # load config file and apply custom cli options
        config = read_json(self.cfg_fname)
        self._config = _update_config(config, options, args)

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])
        timestamp = datetime.now().strftime(r'%m%d_%H%M%S') if timestamp else ''

        exper_name = self.config['name']
        self._save_dir = save_dir / 'models' / exper_name / timestamp
        self._log_dir = save_dir / 'log' / exper_name / timestamp

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # save updated config file to the checkpoint dir
        write_json(self.config, self.save_dir / 'config.json')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    def initialize(self, name, module, *args, **kwargs):
        """
        finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding keyword args given as 'args'.
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def __getitem__(self, name):
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# helper functions used to update config dict with custom cli options
def _update_config(config, options, args):
    for opt in options:
        value = getattr(args, _get_opt_name(opt.flags))
        if value is not None:
            _set_by_path(config, opt.target, value)
    return config

def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
