import os
import sys
import numpy as np
import pandas as pd
from mrcnn import utils
import mrcnn.model as modellib
import cfs_coco_train as coco
from calculate import read_csv

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library

MODEL_DIR = os.path.join(ROOT_DIR, "logs")
config = coco.CocoConfig()

DATASET_DIR = './coco'


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def compute_batch_ap(image_ids):
    APs = []
    for image_id in image_ids:
        # Load image
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id,
                                                                                  use_mini_mask=False)
        # Run object detection
        results = model.detect([image], verbose=0)
        # Compute AP
        r = results[0]
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                                             r['rois'], r['class_ids'], r['scores'], r['masks'])
        print(image_id, AP)
        APs.append(AP)
    return APs


def loop_weight(length, path):
    weights_path = path + ("%04d" % length) + '.h5'
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)
    image_ids = np.random.choice(dataset.image_ids, 30, replace=False, p=None)
    APs = compute_batch_ap(dataset.image_ids)
    print(APs)
    print("mAP @ IoU=50: ", np.mean(APs))
    return np.mean(APs)


def save_data(data, path):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path, index=False, header=False)


config = InferenceConfig()
config.display()
TEST_MODE = "inference"
# with tf.device("/gpu:0"):
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# Load validation dataset
dataset = coco.CocoDataset()
dataset.load_coco(DATASET_DIR, "val2017")
dataset.prepare()
print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

print('\nThe save path for output needs to be the same level as logs folder containing weights.\n')
dataset_name = input('Enter the name of the dataset (e.g., coco): ')
eval_mode = input('Enter the mode for computing AP (first2/full/last5): ')
weights_folder = input('Enter the folder name containing weights: ')
output_path = input('Enter the save path for output (e.g., ../drive/My Drive/custom_s_coco_weights/coco_eval.csv): ')

dir_path = os.path.dirname(output_path)
w_path = dir_path + '/logs/' + weights_folder + '/mask_rcnn_' + dataset_name + '_'

if eval_mode == 'first2':
    weight_amount = 1 + 2
    output = np.zeros(2)
    i = 0
    for index in range(1, weight_amount):
        output[i] = loop_weight(index, w_path)
        i += 1
    save_data(output, output_path)

elif eval_mode == 'full':
    directory = os.path.dirname(output_path)
    filename = (os.path.basename(output_path)).split('.')
    temp_filename = str(filename[0]) + '_temp.' + str(filename[1])
    temp_path = os.path.join(directory, temp_filename)

    m_amount = int(input('Enter the ending model number (e.g., 150 for m150):'))
    output = np.zeros(m_amount)

    is_resumed = input('Do you want to resume computations of mAPs from before? (y/n): ')
    if is_resumed == 'y':
        mAPs_file = input('Enter the path to load the previous computations of mAPs: ')
        previous_mAPs = np.asarray(read_csv(mAPs_file), dtype=np.float64)
        start = previous_mAPs.shape[0]
        previous_mAPs = previous_mAPs.reshape(start)
        output[:start] = previous_mAPs
        start = start + 1
    else:
        start = 1

    end = 1 + m_amount

    i = start - 1
    for index in range(start, end):
        output[i] = loop_weight(index, w_path)
        i += 1

        if (index % 5 == 0) and (index + 1 != end):
            save_data(output, temp_path)

    os.remove(temp_path)
    save_data(output, output_path)

elif eval_mode == 'last5':
    m_num = int(input('Enter the ending model number (e.g., 150 for m150):'))
    weight_amount = 1 + m_num
    start_num = weight_amount - 5
    output = np.zeros(5)
    i = 0
    for index in range(start_num, weight_amount):     # 146-151 = m146 to m150
        output[i] = loop_weight(index, w_path)
        i += 1
    save_data(output, output_path)

else:
    print('Invalid mode entered!')

print('Finish process.')
