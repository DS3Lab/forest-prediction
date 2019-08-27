# Stochastic Adversarial Video Prediction
[[Project Page]](https://alexlee-gk.github.io/video_prediction/) [[Paper]](https://arxiv.org/abs/1804.01523)

TensorFlow implementation for stochastic adversarial video prediction. Given a sequence of initial frames, our model is able to predict future frames of various possible futures. For example, in the next two sequences, we show the ground truth sequence on the left and random predictions of our model on the right. Predicted frames are indicated by the yellow bar at the bottom. For more examples, visit the [project page](https://alexlee-gk.github.io/video_prediction/).

<img src="https://alexlee-gk.github.io/video_prediction/index_files/images/bair_action_free_random_00066_crop.gif" height="96">
<img src="https://alexlee-gk.github.io/video_prediction/index_files/images/bair_action_free_random_00006_crop.gif" height="96">

**Stochastic Adversarial Video Prediction,**  
[Alex X. Lee](https://people.eecs.berkeley.edu/~alexlee_gk/), [Richard Zhang](https://richzhang.github.io/), [Frederik Ebert](https://febert.github.io/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), [Chelsea Finn](https://people.eecs.berkeley.edu/~cbfinn/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/).  
arXiv preprint arXiv:1804.01523, 2018.

An alternative implementation of SAVP is available in the [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor) library.

## Getting Started ###
### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Installation
- Clone this repo:
```bash
git clone -b master --single-branch https://github.com/alexlee-gk/video_prediction.git
cd video_prediction
```
- Install TensorFlow >= 1.9 and dependencies from http://tensorflow.org/
- Install ffmpeg (optional, used to generate GIFs for visualization, e.g. in TensorBoard)
- Install other dependencies
```bash
pip install -r requirements.txt
```
### Miscellaneous installation considerations
- In python >= 3.6, make sure to add the root directory to the `PYTHONPATH`, e.g. `export PYTHONPATH=path/to/video_prediction`.
- For the best speed and experimental results, we recommend using cudnn version 7.3.0.29 and any tensorflow version >= 1.9 and <= 1.12. The final training loss is worse when using cudnn versions 7.3.1.20 or 7.4.1.5, compared to when using versions 7.3.0.29 and below.
- In macOS, make sure that bash >= 4.0 is used (needed for associative arrays in `download_model.sh` script).

### Use a Pre-trained Model
- Download and preprocess a dataset (e.g. `bair`):
```bash
bash data/download_and_preprocess_dataset.sh bair
```
- Download a pre-trained model (e.g. `ours_savp`) for the action-free version of that dataset (i.e. `bair_action_free`):
```bash
bash pretrained_models/download_model.sh bair_action_free ours_savp
```
- Sample predictions from the model:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/generate.py --input_dir data/bair \
  --dataset_hparams sequence_length=30 \
  --checkpoint pretrained_models/bair_action_free/ours_savp \
  --mode test \
  --results_dir results_test_samples/bair_action_free
```
- The predictions are saved as images and GIFs in `results_test_samples/bair_action_free/ours_savp`.
- Evaluate predictions from the model using full-reference metrics:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py --input_dir data/bair \
  --dataset_hparams sequence_length=30 \
  --checkpoint pretrained_models/bair_action_free/ours_savp \
  --mode test \
  --results_dir results_test/bair_action_free
```
- The results are saved in `results_test/bair_action_free/ours_savp`.
- See evaluation details of our experiments in [`scripts/generate_all.sh`](scripts/generate_all.sh) and [`scripts/evaluate_all.sh`](scripts/evaluate_all.sh).

### Model Training
- To train a model, download and preprocess a dataset (e.g. `bair`):
```bash
bash data/download_and_preprocess_dataset.sh bair
```
- Train a model (e.g. our SAVP model on the BAIR action-free robot pushing dataset):
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --input_dir data/bair --dataset bair \
  --model savp --model_hparams_dict hparams/bair_action_free/ours_savp/model_hparams.json \
  --output_dir logs/bair_action_free/ours_savp
```
- To view training and validation information (e.g. loss plots, GIFs of predictions), run `tensorboard --logdir logs/bair_action_free --port 6006` and open http://localhost:6006.
  - Summaries corresponding to the training and validation set are named the same except that the tags of the latter end in "\_1".
  - Summaries corresponding to the validation set with sequences that are longer than the ones used in training end in "\_2", if applicable (i.e. if the dataset's `long_sequence_length` differs from `sequence_length`).
  - Summaries of the metrics over prediction steps are shown as 2D plots in the repurposed PR curves section. To see them, tensorboard needs to be built from source after commenting out two lines from their source code (see [tensorflow/tensorboard#1110](https://github.com/tensorflow/tensorboard/issues/1110)).
  - Summaries with names starting with "eval\_" correspond to the best/average/worst metrics/images out of 100 samples for the stochastic models (as in the paper). The ones starting with "accum\_eval\_" are the same except that they where computed over (roughly) the whole validation set, as opposed to only a single minibatch of the validation set.
- For multi-GPU training, set `CUDA_VISIBLE_DEVICES` to a comma-separated list of devices, e.g. `CUDA_VISIBLE_DEVICES=0,1,2,3`. To use the CPU, set `CUDA_VISIBLE_DEVICES=""`.
- See more training details for other datasets and models in [`scripts/train_all.sh`](scripts/train_all.sh).

### Datasets
Download the datasets using the following script. These datasets are collected by other researchers. Please cite their papers if you use the data.
- Download and preprocess the dataset.
```bash
bash data/download_and_preprocess_dataset.sh dataset_name
```
The `dataset_name` should be one of the following:
- `bair`: [BAIR robot pushing dataset](https://sites.google.com/view/sna-visual-mpc/). [[Citation](data/bibtex/sna.txt)]
- `kth`: [KTH human actions dataset](http://www.nada.kth.se/cvap/actions/). [[Citation](data/bibtex/kth.txt)]

To use a different dataset, preprocess it into TFRecords files and define a class for it. See [`kth_dataset.py`](video_prediction/datasets/kth_dataset.py) for an example where the original dataset is given as videos.

Note: the `bair` dataset is used for both the action-free and action-conditioned experiments. Set the hyperparameter `use_state=True` to use the action-conditioned version of the dataset.

### Models
- Download the pre-trained models using the following script.
```bash
bash pretrained_models/download_model.sh dataset_name model_name
```
The `dataset_name` should be one of the following: `bair_action_free`, `kth`, or `bair`.
The `model_name` should be one of the available pre-trained models:
- `ours_savp`: our complete model, trained with variational and adversarial losses. Also referred to as `ours_vae_gan`.

The following are ablations of our model:
- `ours_gan`: trained with L1 and adversarial loss, with latent variables sampled from the prior at training time.
- `ours_vae`: trained with L1 and KL loss.
- `ours_deterministic`: trained with L1 loss, with no stochastic latent variables.

See [`pretrained_models/download_model.sh`](pretrained_models/download_model.sh) for a complete list of available pre-trained models.

### Model and Training Hyperparameters
The implementation is designed such that each video prediction model defines its architecture and training procedure, and include reasonable hyperparameters as defaults.
Still, a few of the hyperparameters should be overriden for each variant of dataset and model.
The hyperparameters used in our experiments are provided in [`hparams`](hparams) as JSON files, and they can be passed onto the training script with the `--model_hparams_dict` flag.

## Citation

If you find this useful for your research, please use the following.

```
@article{lee2018savp,
  title={Stochastic Adversarial Video Prediction},
  author={Alex X. Lee and Richard Zhang and Frederik Ebert and Pieter Abbeel and Chelsea Finn and Sergey Levine},
  journal={arXiv preprint arXiv:1804.01523},
  year={2018}
}
```
 
