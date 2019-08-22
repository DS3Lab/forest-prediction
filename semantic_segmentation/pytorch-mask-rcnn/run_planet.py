"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import time
import numpy as np
from planet import PlanetDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from planet import PlanetDataset, PlanetDataLoader
from config import PlanetConfig
from base_config import Config
import zipfile
import urllib.request
import shutil

import utils
import model as modellib

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
PLANET_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_planet.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2017"

############################################################
#  COCO Evaluation
############################################################

# def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
#     """Arrange resutls to match COCO specs in http://cocodataset.org/#format
#     """
#     # If no results, return an empty list
#     if rois is None:
#         return []
#
#     results = []
#     for image_id in image_ids:
#         # Loop through detections
#         for i in range(rois.shape[0]):
#             class_id = class_ids[i]
#             score = scores[i]
#             bbox = np.around(rois[i], 1)
#             mask = masks[:, :, i]
#
#             result = {
#                 "image_id": image_id,
#                 "category_id": dataset.get_source_class_id(class_id, "coco"),
#                 "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
#                 "score": score,
#                 "segmentation": maskUtils.encode(np.asfortranarray(mask))
#             }
#             results.append(result)
#     return results

def evaluate_planet(model, dataset, planet, eval_type="bbox", limit=0, image_ids=None):
    pass

def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image])[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on MS Planet")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = PlanetConfig()
    else:
        class InferenceConfig(PlanetConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()
    train_params = config.train_params
    val_params = config.val_params
    # Create model
    model = modellib.MaskRCNN(config=config,
                              model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.model:
        if args.model.lower() == "planet":
            model_path = PLANET_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        # TODO: fill the necessary arguments. See dataloader from UNet
        dataset_train = PlanetDataLoader(train_params, config)
        dataset_val = PlanetDataLoader(val_params, config)

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')

    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = PlanetDataset(val_params)
        print("Running Planet evaluation on {} images.".format(args.limit))
        # evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
        # evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
        evaluate_planet(model, dataset_val, planet, "segm", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
