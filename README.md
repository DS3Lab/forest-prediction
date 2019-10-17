# forest-prediction
🛰Deep learning for deforestation classification and forecasting in satellite imagery

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
$ conda create --name forest-env -r requirements.txt
$ source activate forest-env
```
## Running
You can train the models for semantic segmentation by simply running:
```console
(forest-env) $ cd semantic_segmentation/unet
(forest-env) $ python train.py -c {config_path} -d {gpu_id}
```
This will produce a file having the time in which the script was executed as the folder name.
It will be saved in the "save_dir" value from the JSON file. 
