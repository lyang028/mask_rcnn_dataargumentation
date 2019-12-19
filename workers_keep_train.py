"""
Mask R-CNN
Train on the workers dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 worker.py train --dataset=/path/to/workers/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 worker.py train --dataset=/path/to/workers/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 worker.py train --dataset=/path/to/workers/dataset --weights=imagenet

    # Apply color splash to an image
    python3 worker.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 worker.py splash --weights=last --video=<URL or path to file>
"""
# test
import os
import sys
import random
import math
import json
import datetime
import numpy as np
import skimage.io
import skimage.draw
import time
import tensorflow as tf

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class WorkerConfig(Config):
    """Configuration for training on the action dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "worker"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # Background + action

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 120

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    # VALIDATION_STEPS = 60

    # Length of square anchor side in pixels
    #RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    #RPN_A#NCHOR_SCALE = (4, 8, 16, 32, 64)
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # Skip detections with < 80% confidence
    DETECTION_MIN_CONFIDENCE = 0.8
    DEFAULT_LOGS_DIR = '../drive/My Drive/silhouette_weight/logs'
    # TRAIN_ROIS_PER_IMAGE = 512

############################################################
#  Dataset
############################################################

class WorkerDataset(utils.Dataset):

    def load_worker(self, dataset_dir, subset):
        """Load a subset of the worker dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes.
        self.add_class("action", 1, "other")
        self.add_class("action", 2, "bend")
        self.add_class("action", 3, "squat")
        self.add_class("action", 4, "stand")

        # Train or validation dataset?
        assert subset in ["train", "val","test"]
        dataset_dir = os.path.join(dataset_dir, subset)
        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_export_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinates of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            polygons = [r['shape_attributes'] for r in a['regions']]
            name = [r['region_attributes']['action'] for r in a['regions']]
            name_dict = {"other": 1, "bend": 2, "squat": 3, "stand": 4 }
            name_id = [name_dict[a] for a in name]



            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only manageable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            print(height, width)

            self.add_image(
                "action",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                class_id=name_id,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a worker dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "action":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]

        name_id = image_info["class_id"]
        print(name_id)

        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        class_ids = np.array(name_id, dtype=np.int32)


        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            #print("mask.shape, min(mask),max(mask): {}, {},{}".format(mask.shape, np.min(mask), np.max(mask)))
            #print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr), np.max(rr)))
            #print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc), np.max(cc)))

            ## Note that this modifies the existing array arr, instead of creating a result array
            ## Ref: https://stackoverflow.com/questions/19666626/replace-all-elements-of-python-numpy-array-that-are-greater-than-some-value
            rr[rr > mask.shape[0] - 1] = mask.shape[0] - 1
            cc[cc > mask.shape[1] - 1] = mask.shape[1] - 1

            #print("After fixing the dirt mask, new values:")
            #print("rr.shape, min(rr),max(rr): {}, {},{}".format(rr.shape, np.min(rr), np.max(rr)))
            #print("cc.shape, min(cc),max(cc): {}, {},{}".format(cc.shape, np.min(cc), np.max(cc)))

            mask[rr, cc, i] = 1
            # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        #return (mask.astype(np.bool), class_ids)
        return mask.astype(np.bool), class_ids



    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "action":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model,dataset):
    """Train the model."""
    # Training dataset.
    dataset_train = WorkerDataset()
    dataset_train.load_worker(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = WorkerDataset()
    dataset_val.load_worker(dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    """
    layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
    """
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                layers='heads')

    # print("Fine tune Resnet stage 4 and up")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/10,
    #             epochs=40,
    #             layers='heads')

    # print("train all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE/10,
    #             epochs=60,
    #             layers='all')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.jpg".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################
    # Configurations
config = WorkerConfig()
config.display()
    # Create model
model = modellib.MaskRCNN(mode="training", config=config,model_dir=config.DEFAULT_LOGS_DIR)
    # Load weights

weights_path = COCO_WEIGHTS_PATH
        # Download weights file
if not os.path.exists(weights_path):
    utils.download_trained_weights(weights_path)
model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

#keep train
# weights_path = 'logs/mask_rcnn_worker_0150.h5'
# model.load_weights(weights_path, by_name=True)

# ******************************************* train sihouette
# dataset = 'silhouette320'
# train(model,dataset)

# ******************************************* train stick
dataset = 'stick320'
train(model,dataset)