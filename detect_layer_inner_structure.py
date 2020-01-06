import os
import worker
import mrcnn.model as mrcnn
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import dataReader as dr
from keras import backend as K

ROOT_DIR = os.path.abspath("../../")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

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
#head structure
train_conv_layers = ['fpn_c5p5','fpn_c4p4','fpn_c3p3','fpn_c2p2','fpn_p5','fpn_p2','fpn_p3','fpn_p4']
train_dence_layers = ['mrcnn_mask_conv1','mrcnn_mask_conv2','mrcnn_mask_conv3','mrcnn_mask_conv4',
                      'mrcnn_bbox_fc','mrcnn_mask_deconv','mrcnn_class_logits','mrcnn_mask']
train_normal_layers = ['mrcnn_mask_bn1','mrcnn_mask_bn2','mrcnn_mask_bn3','mrcnn_mask_bn4']

train_rpn_model = 'rpn_model'
#resnet structure
train_resnet_conv = ['conv1',
                     'res2a_branch2a', 'res2a_branch2b','res2a_branch2c',
                     'res2a_branch1',
                     'res2b_branch2a','res2b_branch2b','res2b_branch2c',
                     'res2c_branch2a','res2c_branch2b','res2c_branch2c',

                     'res3a_branch2a','res3a_branch2b','res3a_branch2c',
                     'res3a_branch1',
                     'res3b_branch2a','res3b_branch2b','res3b_branch2c',
                     'res3c_branch2a','res3c_branch2b','res3c_branch2c',

                     'res4a_branch2a','res4a_branch2b','res4a_branch2c',
                     'res4a_branch1',
                     'res4b_branch2a','res4b_branch2b','res4b_branch2c',
                     'res4c_branch2a','res4c_branch2b','res4c_branch2c',
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

def load_weight(path,config):
    # Create model
    model1 = mrcnn.MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
    model1.load_weights(path, by_name=True)
    return model1

def compare_two_denselayer(model1, model2,layer_name):
    weight1 = model1.keras_model.get_layer(layer_name).get_weights()
    weight2 = model2.keras_model.get_layer(layer_name).get_weights()

    arw1 = np.array(weight1)
    arw2 = np.array(weight2)

    if(len(arw1)!=len(arw2)):
        return 0
    compare_two_matrix(arw1[0], arw2[0],layer_name)
    compare_two_array(arw1[1], arw2[1],layer_name)

def compare_two_convlayer(model1, model2,layer_name):
    weight1 = model1.keras_model.get_layer(layer_name).get_weights()
    weight2 = model2.keras_model.get_layer(layer_name).get_weights()
    arw1 = np.array(weight1)
    arw2 = np.array(weight2)
    if(len(arw1)!=len(arw2)):
        return 0
    compare_two_matrix(arw1[0][0][0], arw2[0][0][0],layer_name)
    compare_two_array(arw1[1], arw2[1],layer_name)

def compare_two_normlayer(model1, model2,layer_name):
    weight1 = model1.keras_model.get_layer(layer_name).get_weights()
    weight2 = model2.keras_model.get_layer(layer_name).get_weights()
    arw1 = np.array(weight1)
    arw2 = np.array(weight2)

    if(len(arw1)!=len(arw2)):
        return 0
    compare_two_matrix(arw1, arw2,layer_name,order=1)

def compare_two_array(a1,a2,name = 'default'):
    if(type(a1[0])!=np.float32):
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

def compare_two_matrix(a1,a2,name,order = 0):
    if order ==0:
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
    return  wd
def calculate_wd_bnlayers(m_array1, m_array2, name):
    wd = 0
    weight1 = m_array1.keras_model.get_layer(name).get_weights()
    weight2 = m_array2.keras_model.get_layer(name).get_weights()
    arw1 = np.array(weight1)
    arw2 = np.array(weight2)
    wd = ss.wasserstein_distance(arw1.reshape(arw1.size), arw2.reshape(arw2.size))
    return  wd

def calculate_wd_models_head(model1, model2):
    wd_rpn = calculate_wd_layers(model1, model2, train_rpn_model)
    wd_conv = 0
    for name in train_conv_layers:
        wd_conv = wd_conv + calculate_wd_layers(model1, model2, name)
    wd_dense = 0
    for name in train_dence_layers:
        wd_dense = wd_dense + calculate_wd_layers(model1, model2, name)
    wd_normal = 0
    for name in train_normal_layers:
        wd_normal = wd_normal+ calculate_wd_bnlayers(model1,model2,name)

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

    return wd_conv_array, wd_a ,wd_b, wd_c, wd_r

def sequence_analysis(set_path,path_target):
    file_ls = os.listdir(set_path)
    file_ls.sort()
    wd = []
    counter = 0
    for file_name in file_ls:
        print(file_name)
        weights_path = os.path.join(set_path,file_name)
        model_source = load_weight(weights_path, worker.WorkerConfig())
        model_target = load_weight(path_target, worker.WorkerConfig())
        wd.append(calculate_wd_models_head(model_source, model_target))
        print('**************',len(wd),'**************')
        K.clear_session()
        # if counter>2:
        #     break
        # else:
        #     counter = counter+1
    print('Complete!')
    return wd

def spe_lightweight_sequence_analysis(dataset_path, weight_path,output_path):
    list = np.array(sequence_analysis(dataset_path, weight_path))
    plt.plot(range(len(list[:, 0])), list[:, 0], label='rpn_wd')
    plt.plot(range(len(list[:, 1])), list[:, 1], label='conv_wd')
    plt.plot(range(len(list[:, 2])), list[:, 2], label='dense_wd')
    plt.plot(range(len(list[:, 3])), list[:, 3], label='norm_wd')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'div_wd.png'))
    plt.close()
    sum = list.sum(axis=1)
    plt.plot(range(len(sum)), sum, label='summary_wd')
    plt.savefig(os.path.join(output_path, 'summary_wd.png'))
    plt.close()
    dr.save_data(list, os.path.join(output_path, 'list.csv'))

#...................................................................................


#********************************** sequence compare imagenet coco
# list = np.array(sequence_analysis('logs/Experiments/Sequence_head_compare_imagenet_coco/Set','logs/Experiments/Sequence_head_compare_imagenet_coco/coco.h5'))
# plt.plot(range(len(list[:,0])),list[:,0],label= 'rpn_wd')
# plt.plot(range(len(list[:,1])),list[:,1],label = 'conv_wd')
# plt.plot(range(len(list[:,2])),list[:,2],label = 'dense_wd')
# plt.plot(range(len(list[:,3])),list[:,3],label = 'norm_wd')
# plt.legend()
# plt.savefig('logs/Experiments/Sequence_head_compare_imagenet_coco/div_wd.png')
# plt.close()
# sum = list.sum(axis=1)
# plt.plot(range(len(sum)),sum,label = 'summary_wd')
# plt.savefig('logs/Experiments/Sequence_head_compare_imagenet_coco/summary_wd')
# dr.save_data(list,'logs/Experiments/Sequence_head_compare_imagenet_coco/list.txt')

#******************************compare two resnet 50
# m1 = load_weight('logs/Experiments/compare_resnet_coco/coco.h5',worker.WorkerConfig())
# m2 = load_weight('logs/Experiments/compare_resnet_coco/resnet.h5',worker.WorkerConfig())
#
# wd_array,wd_a,wd_b,wd_c,wd_r = calculate_wd_models_backboon(m1,m2)
# dr.save_data(wd_array, 'logs/Experiments/compare_resnet_coco/result.csv')
# dr.save_data(wd_a, 'logs/Experiments/compare_resnet_coco/result_a.csv')
# dr.save_data(wd_b, 'logs/Experiments/compare_resnet_coco/result_b.csv')
# dr.save_data(wd_c, 'logs/Experiments/compare_resnet_coco/result_c.csv')
# dr.save_data(wd_r, 'logs/Experiments/compare_resnet_coco/result_r.csv')
#
# plt.bar(range(len(wd_array)),wd_array)
# plt.savefig('logs/Experiments/compare_resnet_coco/image.png')
# plt.close()
# plt.bar(range(len(wd_a)),wd_a)
# plt.savefig('logs/Experiments/compare_resnet_coco/image_a.png')
# plt.close()
# plt.bar(range(len(wd_b)),wd_b)
# plt.savefig('logs/Experiments/compare_resnet_coco/image_b.png')
# plt.close()
# plt.bar(range(len(wd_c)),wd_c)
# plt.savefig('logs/Experiments/compare_resnet_coco/image_c.png')
# plt.close()
# plt.bar(range(len(wd_r)),wd_r)
# plt.savefig('logs/Experiments/compare_resnet_coco/image_r.png')
# plt.close()

# *******************************sequence compare imagenet coco
Experiment_path = 'logs/Experiments/Sequence_head_compare_imagenet_coco/'
spe_lightweight_sequence_analysis('../drive/My Drive/silhouette_weight/silhouette_feature')

#*******************************sequence compare coco stickman
Experiment_path = 'logs/Experiments/Sequence_head_compare_coco_stick/'
spe_lightweight_sequence_analysis(Experiment_path)

#********************************sequence compare coco stickman feature
Experiment_path = 'logs/Experiments/Sequence_head_compare_coco_stick_feature/'
spe_lightweight_sequence_analysis(Experiment_path)

#********************************sequence compare coco sil
Experiment_path = 'logs/Experiments/Sequence_head_compare_coco_sil/'
spe_lightweight_sequence_analysis(Experiment_path)

#********************************sequence compare coco sil feature
Experiment_path = 'logs/Experiments/Sequence_head_compare_coco_sil_feature/'
spe_lightweight_sequence_analysis(Experiment_path)

#********************************sequence compare coco mix
Experiment_path = 'logs/Experiments/Sequence_head_compare_coco_mix/'
spe_lightweight_sequence_analysis(Experiment_path)

#********************************sequence compare coco self
Experiment_path = 'logs/Experiments/Sequence_head_compare_coco_self/'
spe_lightweight_sequence_analysis(Experiment_path)

