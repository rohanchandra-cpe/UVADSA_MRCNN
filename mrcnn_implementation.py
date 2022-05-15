from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import mrcnn
from mrcnn.utils import Dataset
from mrcnn import model as modellib, utils
from mrcnn.model import MaskRCNN

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

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from keras.models import load_model
# %matplotlib inline
from os import listdir
from xml.etree import ElementTree
import skimage.draw

ROOT_DIR = "./"
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class myMaskRCNNConfig(Config):
    # give the configuration a recognizable name
    NAME = "MaskRCNN_config"
 
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
 
    # number of classes (we would normally add +1 for the background)
    NUM_CLASSES = 1+8
   
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 131
    
    # Learning rate
    LEARNING_RATE=0.006
    
    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # setting Max ground truth instances
    MAX_GT_INSTANCES=10

class RavenDataset(Dataset):
    def load_dataset(self, dataset_dir, subset):
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

        # training, validation, or testing
        assert subset in ['train', 'test', 'validation']
        images_dataset_dir = os.path.join(dataset_dir, subset + "/images/")

        for dirpath, dirnames, files in os.walk(images_dataset_dir):
            for file_name in files: # we have to self.add image for each file_name
                image_file_name = file_name
                json_file_name = file_name[0:len(file_name) - len(".png")] + ".json"
                annotations = json.load(open(os.path.join(dataset_dir, subset + "/json/" + json_file_name)))

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
                    "object",
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
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
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
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def train(model):
    # Train the model
    train_set = RavenDataset()
    train_set.load_dataset("./", 'train') # path to training data here
    train_set.prepare()

    val_set = RavenDataset()
    val_set.load_dataset("./", 'validation') # path to val data here
    val_set.prepare()

    # More Step to Come Later!
    print("Training network heads")
    model.train(train_set, val_set,
                learning_rate = model.config.LEARNING_RATE,
                epochs = 10,
                layers = 'heads')
                
def main():
    config = myMaskRCNNConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                                    model_dir=DEFAULT_LOGS_DIR)

    weights_path = COCO_WEIGHTS_PATH
            # Download weights file
    if not os.path.exists(weights_path):
        utils.download_trained_weights(weights_path)

    model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
    print("Commence Training!")
    train(model)

if __name__ == "__main__":
    main()

# requirements file, make a github repo with info on how to set it up and run it
# JIGSAW dataset, got it from Cogito, Ian has information about that 
# Focus on the technical approach 
# Ask Kay about using machines to train the MRCNN
# I need a good GPU

# Report should be in by the lsat day of final exams, one good version ~ May 13th, should be a first draft. We can revise it later 

# IBIS