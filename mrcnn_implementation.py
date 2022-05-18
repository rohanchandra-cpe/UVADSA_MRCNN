import mrcnn
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize as visualize
from mrcnn.visualize import display_images
from mrcnn.visualize import display_instances

from mrcnn.utils import Dataset
from mrcnn import model as modellib, utils
from mrcnn.model import MaskRCNN, log

import numpy as np
from numpy import zeros
from numpy import asarray

import colorsys
import argparse
import imutils
import random
# import cv2
import os
import time
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import tensorflow as tf
import sys
import math
import re
from keras.models import load_model
from os import listdir
from xml.etree import ElementTree
import skimage.draw

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join("./", "logs")

class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 5
    
    # Learning rate
    LEARNING_RATE=0.006
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10

    # Number of classes
    NUM_CLASSES = 1 + 8

# class InferenceConfig(Config):
#     # Run detection on one image at a time
#     NAME = "INFERENCE_config"
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1
#     DETECTION_MIN_CONFIDENCE= 0.7
#     NUM_CLASSES = 1 + 8

class JigsawDataset(Dataset):
    def load_dataset(self, dataset_dir):
        # 8 classes for the knot_tying dataset
        # Hardcoded for now, but will change later
        self.add_class("dataset", 1, "Needle Tip") # Do I have more than 1 class?
        self.add_class("dataset", 2, "Needle End")
        self.add_class("dataset", 3, "Left Grasper")
        self.add_class("dataset", 4, "Right Grasper")
        self.add_class("dataset", 5, "Top Left Thread")
        self.add_class("dataset", 6, "Top Right Thread")
        self.add_class("dataset", 7, "Bottom Left Thread")
        self.add_class("dataset", 8, "Bottom Right Thread")

        images_dataset_dir = dataset_dir + "/images/"

        for dirpath, dirnames, files in os.walk(images_dataset_dir):
            for file_name in files: # we have to self.add image for each file_name
                image_file_name = file_name
                json_file_name = file_name[0:len(file_name) - len(".png")] + ".json"
                annotations = json.load(open(dataset_dir + "/json/" + json_file_name))

                polygons = []
                num_ids = []
                name_dict = {
                    "Needle Tip": 1,
                    "Needle End": 2,
                    "Left Grasper": 3,
                    "Right Grasper": 4,
                    "Top Left Thread": 5,
                    "Top Right Thread": 6,
                    "Bottom Left Thread": 7,
                    "Bottom Right Thread": 8
                }
                
                for a in annotations["instances"]: #For every instance in the image's JSON file
                    shape_attributes = {}
                    points = a["points"]
                    class_name = a["className"]
                    x = []
                    y = []
                    for i in range(len(points)):
                        if(i % 2 == 0):
                            x.append(points[i])
                        else:
                            y.append(points[i])
                    shape_attributes['all_points_x'] = x
                    shape_attributes['all_points_y'] = y
                    shape_attributes['name'] = a["type"]
                    polygons.append(shape_attributes)
                    num_ids.append(name_dict[class_name])
                
                self.add_image(
                    "dataset",
                    image_id = dirpath + file_name,
                    path = dirpath + file_name,
                    width = annotations["metadata"]["width"], #640
                    height = annotations["metadata"]["height"], #480
                    polygons = polygons,
                    num_ids = num_ids
                )
        return 0

    def load_mask(self, image_id):        
        #convert polygons to a bit mask of shape
        image_info = self.image_info[image_id]
        if image_info["source"] != "dataset":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "dataset":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
        	mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "dataset":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def visualize_images():
    SURGERY_WEIGHTS_PATH = "./logs/maskrcnn_config20220516T2057/mask_rcnn_maskrcnn_config_0002.h5"
    DEVICE = "/cpu:0"

    config = InferenceConfig()
    config.display()

    # Load validation dataset
    val_set = JigsawDataset()
    val_set.load_dataset("./", 'val_small') # path to val data here
    val_set.prepare()

    print("Images: {}\nClasses: {}".format(len(val_set.image_ids), val_set.class_names))

    with(tf.device(DEVICE)):
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)

    # Load weights
    print("Loading weights ", SURGERY_WEIGHTS_PATH)
    model.load_weights(SURGERY_WEIGHTS_PATH, by_name=True)

    image_id = random.choice(val_set.image_ids)
    image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(val_set, config, image_id, use_mini_mask=False)
    info = val_set.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,val_set.image_reference(image_id)))

    # Run object detection
    results = model.detect([image], verbose=1)

    # Display results
    ax = get_ax(1)
    r = results[0]
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            val_set.class_names, r['scores'], ax=ax,
                            title="Predictions")
    visualize.display_images(image)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

def train(model):
    print("############################")
    print(args.train_dataset)
    print(args.val_dataset)
    print("############################")
    train_set = JigsawDataset()
    train_set.load_dataset(args.train_dataset) # path to training data here
    train_set.prepare()

    val_set = JigsawDataset()
    val_set.load_dataset(args.val_dataset) # path to val data here
    val_set.prepare()

    print("Training network heads")
    model.train(train_set, val_set,
                learning_rate = model.config.LEARNING_RATE, epochs = 2, layers = 'heads')

#############################
#   Main Method
#############################
if __name__ == '__main__':
    ROOT_DIR = "./"
    COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect images from the JIGSAW Dataset.')
    parser.add_argument("command",
                        help="'train' or 'visualize_images'")
    parser.add_argument('--train_dataset', required = True,
                        help='Directory of the training dataset')
    parser.add_argument('--val_dataset', required = True,
                        help='Directory of the validation dataset')
    parser.add_argument('--weights', required=True,
                        help="Path to weights .h5 file or 'coco'")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.train_dataset, "Argument --train_dataset is required for training"
        assert args.val_dataset, "Argument --val_dataset is required for training"

    print("Weights: ", args.weights)
    print("Training Dataset: ", args.train_dataset)
    print("Validation Dataset: ", args.val_dataset)

    # Configurations
    if args.command == "train":
        config = myMaskRCNNConfig()
    else:
        class InferenceConfig(SurgeryConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            NAME = "INFERENCE_config"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE= 0.7
            NUM_CLASSES = 1 + 8
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            print("------------------------------------------------------------------")
            utils.download_trained_weights(weights_path)
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))