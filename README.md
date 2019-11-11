# forest-prediction
🛰Deep learning for deforestation classification and forecasting in satellite imagery

[![](https://tinyurl.com/greenai-pledge)](https://github.com/daviddao/green-ai)

## Overview
In this repository we provide implementations for:
1. Data scraping (Tile services and Google Earth Engine)
2. Forest prediction (Semantic Segmentation)
3. Video prediction (Lee et al, 2018)
4. Image to image translation (Isola et al, 2017)

## Installation
```console
$ git clone https://github.com/DS3Lab/forest-prediction.git
$ cd forest-prediction
$ conda create --name forest-env python=3.7
$ ./install.sh
$ source activate forest-env
```
## Running
You can train the models for semantic segmentation by simply running:
```console
(forest-env) $ cd semantic_segmentation/unet
(forest-env) $ python train.py -c {config_path} -d {gpu_id}
```
For multi-GPU training, set gpu_id to a comma-separated list of devices, e.g. 
-d 0,1,2,3,4
This will produce a file having the time in which the script was executed as the folder name.
It will be saved in the "save_dir" value from the JSON file, under "trainer". Under save_dir, it will create
a log file, where you can check Tensorboard, and a model file, where the model is going to be stored.

## Testing
You can test the models for semantic segmentation by running:
```console
(forest-env) $ python simple_test.py -r {model_saved_path/model.pth} -d {gpu_id}
```
It will run the predictions and save the corresponding outputs in model_saved_path. To keep an order of the images, set both `batch_size` and `num_workers` to 1.

## Configuration
You can change the type of model used, and its configuration by altering (or creating) a config.json file. 

### Structure of `config.json`
The fields of the config file are self explanatory. We explain the most important ones.
* `name`: indicates the name of the experiment. It is the folder in which both the training logs and models are going to be stored
* `n_gpu`: for multi-GPU training, it is necessary to specify how many gpus it is going to use. For instance, if the user specifies `-d 0,1`, in order to use both gpus `n_gpu` needs to be set up to 2. If it is set up to 1, it will only use gpu 0, if it is set up to a number higher than 2, then it will yield an error.
* `arch`: it specifies the model that will be used for training/testing purposes.
* `data_loader_train` and `data_loader_val`: data loaders for training and validation purposes. For testing, only            `data_loader_val` is used. 
    

