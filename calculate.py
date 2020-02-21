import sys
import os
import cfs_coco_train as coco
import numpy as np
import pandas as pd
import csv
import shutil
import matplotlib.pyplot as plt
import scipy.stats as ss
from keras import backend as K
from mrcnn.config import Config
import mrcnn.model as mlib
import pathlib

#
# mrcnn_class_logits
# mrcnn_class
# mrcnn_bbox_fc
# mrcnn_bbox
# roi_align_classifier
# mrcnn_class_conv1
# mrcnn_class_bn1
# mrcnn_class_conv2
# mrcnn_class_bn2
# pool_squeeze
#

# head structure
train_conv_layers = ['fpn_c5p5', 'fpn_c4p4', 'fpn_c3p3', 'fpn_c2p2', 'fpn_p5', 'fpn_p2', 'fpn_p3', 'fpn_p4']
train_dense_layers = ['mrcnn_mask_conv1', 'mrcnn_mask_conv2', 'mrcnn_mask_conv3', 'mrcnn_mask_conv4',
                      'mrcnn_bbox_fc', 'mrcnn_mask_deconv', 'mrcnn_class_logits', 'mrcnn_mask']
train_normal_layers = ['mrcnn_mask_bn1', 'mrcnn_mask_bn2', 'mrcnn_mask_bn3', 'mrcnn_mask_bn4']
train_rpn_model = 'rpn_model'

# resnet structure
train_resnet_conv = ['conv1',
                     'res2a_branch2a', 'res2a_branch2b', 'res2a_branch2c',
                     'res2a_branch1',
                     'res2b_branch2a', 'res2b_branch2b', 'res2b_branch2c',
                     'res2c_branch2a', 'res2c_branch2b', 'res2c_branch2c',

                     'res3a_branch2a', 'res3a_branch2b', 'res3a_branch2c',
                     'res3a_branch1',
                     'res3b_branch2a', 'res3b_branch2b', 'res3b_branch2c',
                     'res3c_branch2a', 'res3c_branch2b', 'res3c_branch2c',

                     'res4a_branch2a', 'res4a_branch2b', 'res4a_branch2c',
                     'res4a_branch1',
                     'res4b_branch2a', 'res4b_branch2b', 'res4b_branch2c',
                     'res4c_branch2a', 'res4c_branch2b', 'res4c_branch2c',
                     'res4d_branch2a', 'res4d_branch2b', 'res4d_branch2c',
                     'res4e_branch2a', 'res4e_branch2b', 'res4e_branch2c',
                     'res4f_branch2a', 'res4f_branch2b', 'res4f_branch2c',
                     'res4g_branch2a', 'res4g_branch2b', 'res4g_branch2c',
                     'res4h_branch2a', 'res4h_branch2b', 'res4h_branch2c',
                     'res4i_branch2a', 'res4i_branch2b', 'res4i_branch2c',
                     'res4j_branch2a', 'res4j_branch2b', 'res4j_branch2c',
                     'res4k_branch2a', 'res4k_branch2b', 'res4k_branch2c',
                     'res4l_branch2a', 'res4l_branch2b', 'res4l_branch2c',
                     'res4m_branch2a', 'res4m_branch2b', 'res4m_branch2c',
                     'res4n_branch2a', 'res4n_branch2b', 'res4n_branch2c',
                     'res4o_branch2a', 'res4o_branch2b', 'res4o_branch2c',
                     'res4p_branch2a', 'res4p_branch2b', 'res4p_branch2c',
                     'res4q_branch2a', 'res4q_branch2b', 'res4q_branch2c',
                     'res4r_branch2a', 'res4r_branch2b', 'res4r_branch2c',
                     'res4s_branch2a', 'res4s_branch2b', 'res4s_branch2c',
                     'res4t_branch2a', 'res4t_branch2b', 'res4t_branch2c',
                     'res4u_branch2a', 'res4u_branch2b', 'res4u_branch2c',
                     'res4v_branch2a', 'res4v_branch2b', 'res4v_branch2c',
                     'res4w_branch2a', 'res4w_branch2b', 'res4w_branch2c',

                     'res5a_branch2a', 'res5a_branch2b', 'res5a_branch2c',
                     'res5a_branch1',
                     'res5b_branch2a', 'res5b_branch2b', 'res5b_branch2c',
                     'res5c_branch2a', 'res5c_branch2b', 'res5c_branch2c',
                     ]
train_resnet_conv_a = [
    'res2a_branch2a',
    'res2b_branch2a',
    'res2c_branch2a',
    'res3a_branch2a',
    'res3b_branch2a',
    'res3c_branch2a',
    'res4a_branch2a',

    'res4b_branch2a',
    'res4c_branch2a',
    'res4d_branch2a',
    'res4e_branch2a',
    'res4f_branch2a',
    'res4g_branch2a',
    'res4h_branch2a',
    'res4i_branch2a',
    'res4j_branch2a',
    'res4k_branch2a',
    'res4l_branch2a',
    'res4m_branch2a',
    'res4n_branch2a',
    'res4o_branch2a',
    'res4p_branch2a',
    'res4q_branch2a',
    'res4r_branch2a',
    'res4s_branch2a',
    'res4t_branch2a',
    'res4u_branch2a',
    'res4v_branch2a',
    'res4w_branch2a',

    'res5a_branch2a',
    'res5b_branch2a',
    'res5c_branch2a',
]
train_resnet_conv_b = [
    'res2a_branch2b',
    'res2b_branch2b',
    'res2c_branch2b',

    'res3a_branch2b',
    'res3b_branch2b',
    'res3c_branch2b',
    'res4a_branch2b',
    'res4b_branch2b',
    'res4c_branch2b',
    'res4d_branch2b',
    'res4e_branch2b',
    'res4f_branch2b',
    'res4g_branch2b',
    'res4h_branch2b',
    'res4i_branch2b',
    'res4j_branch2b',
    'res4k_branch2b',
    'res4l_branch2b',
    'res4m_branch2b',
    'res4n_branch2b',
    'res4o_branch2b',
    'res4p_branch2b',
    'res4q_branch2b',
    'res4r_branch2b',
    'res4s_branch2b',
    'res4t_branch2b',
    'res4u_branch2b',
    'res4v_branch2b',
    'res4w_branch2b',

    'res5a_branch2b',
    'res5b_branch2b',
    'res5c_branch2b',
]
train_resnet_conv_c = [
    'res2a_branch2c',
    'res2b_branch2c',
    'res2c_branch2c',

    'res3a_branch2c',
    'res3b_branch2c',
    'res3c_branch2c',

    'res4a_branch2c',
    'res4b_branch2c',
    'res4c_branch2c',
    'res4d_branch2c',
    'res4e_branch2c',
    'res4f_branch2c',
    'res4g_branch2c',
    'res4h_branch2c',
    'res4i_branch2c',
    'res4j_branch2c',
    'res4k_branch2c',
    'res4l_branch2c',
    'res4m_branch2c',
    'res4n_branch2c',
    'res4o_branch2c',
    'res4p_branch2c',
    'res4q_branch2c',
    'res4r_branch2c',
    'res4s_branch2c',
    'res4t_branch2c',
    'res4u_branch2c',
    'res4v_branch2c',
    'res4w_branch2c',

    'res5a_branch2c',
    'res5b_branch2c',
    'res5c_branch2c',
]
train_resnet_conv_r = ['conv1',
                       'res2a_branch1',
                       'res3a_branch1',
                       'res4a_branch1',
                       'res5a_branch1',
                       ]

# coco classes ids (use dog, cake, bed for base footprints)
coco_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports', 'ball', 'kite',
                'baseball', 'bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
                'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                'laptop', 'mouse', 'remote keyboard', 'cell phone', 'microwave oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock',
                'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

DEFAULT_LOGS_DIR = coco.CocoConfig().DEFAULT_LOGS_DIR
SAVE_MODEL_DIR = os.path.join(DEFAULT_LOGS_DIR, 'exp_generated')


def save_data(data, path):
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(path, index=False, header=False)


def read_csv(path):
    csv_file = open(path, "r")
    reader = csv.reader(csv_file)
    result = []
    for item in reader:
        result.append(item)
    csv_file.close()
    return result


def load_weight(path, config):
    # path to save model trained that will to be deleted
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)

    # Create model then load weight
    model1 = mlib.MaskRCNN(mode="training", config=config, model_dir=SAVE_MODEL_DIR)

    # changes made adding exclude arguments for samples mode when load and train
    # model1.load_weights(path, by_name=True, exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', 'mrcnn_mask'])
    model1.load_weights(path, by_name=True)
    return model1


def compare_two_denselayer(model1, model2, layer_name):
    weight1 = model1.keras_model.get_layer(layer_name).get_weights()
    weight2 = model2.keras_model.get_layer(layer_name).get_weights()

    arw1 = np.array(weight1)
    arw2 = np.array(weight2)

    if (len(arw1) != len(arw2)):
        return 0
    compare_two_matrix(arw1[0], arw2[0], layer_name)
    compare_two_array(arw1[1], arw2[1], layer_name)


def compare_two_convlayer(model1, model2, layer_name):
    weight1 = model1.keras_model.get_layer(layer_name).get_weights()
    weight2 = model2.keras_model.get_layer(layer_name).get_weights()
    arw1 = np.array(weight1)
    arw2 = np.array(weight2)
    if (len(arw1) != len(arw2)):
        return 0
    compare_two_matrix(arw1[0][0][0], arw2[0][0][0], layer_name)
    compare_two_array(arw1[1], arw2[1], layer_name)


def compare_two_normlayer(model1, model2, layer_name):
    weight1 = model1.keras_model.get_layer(layer_name).get_weights()
    weight2 = model2.keras_model.get_layer(layer_name).get_weights()
    arw1 = np.array(weight1)
    arw2 = np.array(weight2)

    if (len(arw1) != len(arw2)):
        return 0
    compare_two_matrix(arw1, arw2, layer_name, order=1)


def compare_two_array(a1, a2, name='default'):
    if (type(a1[0]) != np.float32):
        return
    d = a1 - a2
    if (sum(d) < 0.0001):
        return
    print('Different array')
    x = np.array(range(len(a1)))
    width = 0.35
    plt.bar(x, a1, width=width, label='a1', fc='y')
    plt.bar(x + width, a2, width=width, label='a2', fc='r')
    plt.legend()
    plt.savefig(name)


def compare_two_matrix(a1, a2, name, order=0):
    if order == 0:
        for i in range(len(a1[0])):
            compare_two_array(a1[:, i], a2[:, i], name + str(i))
    else:
        for i in range(len(a1)):
            compare_two_array(a1[i], a2[i], name + str(i))


def calculate_wd_layers(m_array1, m_array2, name):
    wd = 0
    weight1 = m_array1.keras_model.get_layer(name).get_weights()
    weight2 = m_array2.keras_model.get_layer(name).get_weights()
    arw1 = np.array(weight1)
    arw2 = np.array(weight2)
    for i in range(arw1.size):
        a1 = arw1[i].reshape(arw1[i].size)
        a2 = arw2[i].reshape(arw2[i].size)
        wd = wd + ss.wasserstein_distance(a1, a2)
    return wd


def calculate_wd_bnlayers(m_array1, m_array2, name):
    wd = 0
    weight1 = m_array1.keras_model.get_layer(name).get_weights()
    weight2 = m_array2.keras_model.get_layer(name).get_weights()
    arw1 = np.array(weight1)
    arw2 = np.array(weight2)
    wd = ss.wasserstein_distance(arw1.reshape(arw1.size), arw2.reshape(arw2.size))
    return wd


def calculate_wd_models_head(model1, model2):
    wd_rpn = calculate_wd_layers(model1, model2, train_rpn_model)
    wd_conv = 0
    for name in train_conv_layers:
        wd_conv = wd_conv + calculate_wd_layers(model1, model2, name)
    wd_dense = 0
    for name in train_dense_layers:
        wd_dense = wd_dense + calculate_wd_layers(model1, model2, name)
    wd_normal = 0
    for name in train_normal_layers:
        wd_normal = wd_normal + calculate_wd_bnlayers(model1, model2, name)

    return wd_rpn, wd_conv, wd_dense, wd_normal


def calculate_wd_models_backboon(model1, model2):
    wd_conv_array = []
    wd_a = []
    wd_b = []
    wd_c = []
    wd_r = []
    for name in train_resnet_conv:
        wd_conv_array.append(calculate_wd_layers(model1, model2, name))
    for name in train_resnet_conv_a:
        wd_a.append(calculate_wd_layers(model1, model2, name))
    for name in train_resnet_conv_b:
        wd_b.append(calculate_wd_layers(model1, model2, name))
    for name in train_resnet_conv_c:
        wd_c.append(calculate_wd_layers(model1, model2, name))
    for name in train_resnet_conv_r:
        wd_r.append(calculate_wd_layers(model1, model2, name))

    return wd_conv_array, wd_a, wd_b, wd_c, wd_r


def calculate_wd_models_all(model1, model2):
    wd_rpn = calculate_wd_layers(model1, model2, train_rpn_model)
    wd_conv = 0
    wd_dense = 0
    wd_normal = 0
    for name in train_conv_layers:
        wd_conv = wd_conv + calculate_wd_layers(model1, model2, name)
    for name in train_dense_layers:
        wd_dense = wd_dense + calculate_wd_layers(model1, model2, name)
    for name in train_normal_layers:
        wd_normal = wd_normal + calculate_wd_bnlayers(model1, model2, name)

    wd_resnet_conv = 0
    for name in train_resnet_conv:
        wd_resnet_conv = wd_resnet_conv + calculate_wd_layers(model1, model2, name)

    final_wd = wd_rpn + wd_conv + wd_dense + wd_normal + wd_resnet_conv

    return final_wd


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Construct the dataset used for experiment.')
    parser.add_argument('--mode', required=True,
                        metavar="<original|samples>",
                        help="Kind of experiment to be conducted.")
    parser.add_argument('--dataPath', required=True,
                        metavar="<folder_name_of_weights>",
                        help="Folder name in logs containing the weights")
    parser.add_argument('--modelAmount', required=False,
                        default=150,
                        metavar="<m_amount>",
                        help="Amount of models in weights folder (e.g., 150 for m1 to m150) (default is 150)",
                        type=int)
    args = parser.parse_args()

    mode = args.mode
    assert mode in ['original', 'samples']
    folder_name = args.dataPath
    m_amount = args.modelAmount

    # variables
    decimal_place = 8
    model_name = 'mask_rcnn_coco_'
    save_csv_dir = '../drive/My Drive/cfs_' + folder_name + '/'
    # os.makedirs(save_csv_dir, exist_ok=True)
    pathlib.Path(save_csv_dir).mkdir(parents=True, exist_ok=True)
    eps = 1

    model_path = os.path.join(DEFAULT_LOGS_DIR, folder_name)
    end_model = (str(m_amount).zfill(4)) + '.h5'

    # calculate wd, v for original footprint
    if mode == 'original':
        start = int(input('Start or continue from which model for wd (e.g., 1 for m1)? Enter: '))

        save_wd_path = save_csv_dir + 'wd.csv'
        wd = []
        temp_save_wd = save_csv_dir + 'wd_temp.csv'
        if start != 1:
            temp_size = start - 1
            previous = np.asarray(read_csv(temp_save_wd), dtype=np.float64)
            previous = np.reshape(previous, temp_size)
            for index in range(temp_size):
                wd.append(previous[index])

        last_model_fullname = model_name + end_model
        last_model_path = os.path.join(model_path, last_model_fullname)

        # calculate wd
        for i in range(start, m_amount):
            last_model = load_weight(last_model_path, coco.CocoConfig())

            model_i = (str(i).zfill(4)) + '.h5'
            model_i_fullname = model_name + model_i
            current_model_path = os.path.join(model_path, model_i_fullname)
            current_model = load_weight(current_model_path, coco.CocoConfig())

            current_wd = calculate_wd_models_all(last_model, current_model)
            current_wd = round(current_wd, decimal_place)
            wd.append(current_wd)
            print('wd' + str(i) + ' is computed')
            K.clear_session()

            if i % 5 == 0:
                temp_wd = np.asarray(wd, dtype=np.float64)
                np.savetxt(temp_save_wd, temp_wd, delimiter=',', fmt='%f')

        # save wd
        wd = np.asarray(wd, dtype=np.float64)
        np.savetxt(save_wd_path, wd, delimiter=',', fmt='%f')
        os.remove(temp_save_wd)
        print('The wasserstein distances are computed and saved.\n')
        # NOTE: wd[0] is the wd between m1 and end model (m150) ... wd[148] = wd between m149 and m150

        # velocity
        epoch = 1
        save_v_path = save_csv_dir + 'v.csv'
        v = []

        for i in range((m_amount - 1)):
            if i == 0:
                delta_wd = wd[i] - wd[i + 1]
            elif i != (m_amount - 2):
                delta_wd = wd[i - 1] - wd[i + 1]
            else:
                delta_wd = wd[i]

            current_v = delta_wd / (2 * epoch)
            current_v = round(current_v, decimal_place)
            v.append(current_v)

        v = np.asarray(v, dtype=np.float64)
        np.savetxt(save_v_path, v, delimiter=',', fmt='%f')
        print('The velocities are computed and saved.\n')

    # calculate wdp, vp with different samples
    elif mode == 'samples':
        start = int(input('Start or continue from which model for wdp? Enter: '))
        lr = float(input('Enter the learning rate of models trained: '))
        set_num = input('Which experiment set number is this (1/2/3.1 to 3.2/4)? Enter: ')

        save_wdp_path = save_csv_dir + 'wdp' + set_num + '.csv'
        wdp = []

        temp_save_wdp = save_csv_dir + 'wdp_temp.csv'
        if start != 1:
            temp_size = start - 1
            previous = np.asarray(read_csv(temp_save_wdp), dtype=np.float64)
            previous = np.reshape(previous, temp_size)
            for index in range(temp_size):
                wdp.append(previous[index])

        last_model_fullname = model_name + end_model
        last_model_path = os.path.join(model_path, last_model_fullname)

        config = coco.CocoConfig()
        config.LEARNING_RATE = lr
        coco_path = './coco'
        # choose data depending on set_num
        if set_num == '1':
            dataset_train = coco.CocoDataset()
            dataset_train.load_coco(coco_path, "samples")
            dataset_train.prepare()
        elif set_num == '2':
            dataset_train = coco.CocoDataset()
            dataset_train.load_coco(coco_path, "dark")
            dataset_train.prepare()
        elif set_num == '3.1':
            dataset_train = coco.CocoDataset()
            dataset_train.load_coco(coco_path, "samples", shift='1')
            dataset_train.prepare()
        elif set_num == '3.2':
            dataset_train = coco.CocoDataset()
            dataset_train.load_coco(coco_path, "samples", shift='2')
            dataset_train.prepare()
        elif set_num == '4':
            dataset_train = coco.CocoDataset()
            dataset_train.load_coco(coco_path, "samples", shift='random')
            dataset_train.prepare()
        else:
            print('****************************************')
            print('Invalid set number has been entered!')
            print('****************************************')
            exit(1)

        dataset_val = coco.CocoDataset()
        dataset_val.load_coco(coco_path, "val2017")
        dataset_val.prepare()

        # calculate wdp and save wdp every 5 times
        for i in range(start, (m_amount + 1)):
            last_model = load_weight(last_model_path, coco.CocoConfig())

            model_i = (str(i).zfill(4)) + '.h5'
            model_i_fullname = model_name + model_i
            current_model_path = os.path.join(model_path, model_i_fullname)
            current_model = load_weight(current_model_path, coco.CocoConfig())
            current_model.train(dataset_train, dataset_val,
                                learning_rate=config.LEARNING_RATE,
                                epochs=eps,
                                layers='all')

            current_wd_prime = calculate_wd_models_all(last_model, current_model)
            current_wd_prime = round(current_wd_prime, decimal_place)
            wdp.append(current_wd_prime)
            print('wdp' + str(i) + ' is computed')
            K.clear_session()
            shutil.rmtree(SAVE_MODEL_DIR, ignore_errors=True)

            if (i % 5 == 0) and (i != m_amount):
                temp_wdp = np.asarray(wdp, dtype=np.float64)
                np.savetxt(temp_save_wdp, temp_wdp, delimiter=',', fmt='%f')

        wdp = np.asarray(wdp, dtype=np.float64)
        np.savetxt(save_wdp_path, wdp, delimiter=',', fmt='%f')
        os.remove(temp_save_wdp)
        print('The wd prime of each model is calculated and saved.\n')
        # NOTE: wdp[0] is the wd between m1' and end model (m150) ... wdp[149] = wd between m150' and m150

        # vp
        load_wd_path = save_csv_dir + 'wd.csv'
        wd = np.asarray(read_csv(load_wd_path), dtype=np.float64)
        wd = np.reshape(wd, (m_amount - 1))
        save_vp_path = save_csv_dir + 'vp' + set_num + '.csv'
        vp = []

        for i in range(m_amount):
            if i == 0:
                delta_wd_prime = wd[i] - wdp[i + 1]
            elif i != (m_amount - 1):
                delta_wd_prime = wd[i - 1] - wdp[i + 1]
            else:
                delta_wd_prime = wdp[i]

            current_v_prime = delta_wd_prime / (2 * eps)
            current_v_prime = round(current_v_prime, decimal_place)
            vp.append(current_v_prime)

        vp = np.asarray(vp, dtype=np.float64)
        np.savetxt(save_vp_path, vp, delimiter=',', fmt='%f')
        print('The velocity prime of each model is calculated and saved.\n')

    else:
        print('****************************************')
        print('Mode entered is out of scope!')
        print('****************************************')

    print('\nFinish the experiment.')
