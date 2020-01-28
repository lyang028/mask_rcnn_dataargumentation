import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as skm
import dataReader as dr

def relabel_10_even(label):
    output = np.zeros([len(label)],dtype=int)
    for i in range(len(output)):
        output[i] = int(label[i,1]/10)
    return  output

def relabel_2(label):
    output = np.zeros([len(label)],dtype=int)
    for i in range(len(output)):
        if label[i,1]>50:
            output[i] = 1
    return  output

def recalculate_coordinate(coordinate):
    input1 = tf.constant(coordinate[:,0])
    input2 = tf.constant(coordinate[:,1])

    output_tf1 = tf.sigmoid(input1)
    output_tf2 = tf.sigmoid(input2)

    sess = tf.Session()
    return [sess.run(output_tf1), sess.run(output_tf2)]

def cluster(data, label):
    output = np.zeros([11,2])
    # print(output[8, 1])
    # print(data[0][0])
    count = np.zeros([11])
    for i in range(len(label)):
        # print(output[label[i], 0])
        output[label[i], 0] = output[label[i],0] + data[0][i]
        output[label[i], 1] = output[label[i],1] + data[1][i]
        count[label[i]] = count[label[i]]+1
    for i in range(len(output)):
        output[i] = output[i]/count[i]

    return output

def calculate_silhouse_score(path, dataamount,re_type = 10,recoordinate = True):
    output = np.zeros(dataamount)
    for i in range(dataamount):
        vis_path = path+'Data2D_'+ str(i)+'.csv'

        label_path = path+'Labels.csv'
        mat = pd.read_csv(vis_path, header=None)
        label = pd.read_csv(label_path, header=None)
        mat_list = mat.to_numpy()
        label_list = label.to_numpy()

        if re_type ==10:
            re_label = relabel_10_even(label_list)
        else:
            re_label = relabel_2(label_list)
        re_coord = recalculate_coordinate(mat_list)
        sil = skm.silhouette_score(np.transpose(re_coord),re_label, metric='euclidean')
        output[i] = sil
    return output
def draw_vis_image(labels,points,pic_name):
    csvFile = dr.read_csv(labels)

    y_data = np.array(csvFile, np.float32)
    y = y_data[:, 1]
    coordinate = dr.read_csv(points)
    encoded_data_mean = np.array(coordinate, np.float32)
    encoded_data_output = tf.nn.sigmoid(encoded_data_mean)
    sess = tf.Session()
    code_output = sess.run(encoded_data_output)

    # draw image
    fig = plt.figure(figsize=(6, 6))

    plt.scatter(code_output[:, 0], code_output[:, 1], c=y)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    bar = plt.colorbar()
    bar.ax.tick_params(labelsize = 15)
    plt.savefig(pic_name)
def draw_curve_image(curves,pic_name,ratio = 1):
    csvFile = dr.read_csv(curves)
    data = np.array(csvFile, np.float32)

    # draw image
    fig = plt.figure(figsize=(6, 6))
    if ratio !=1:
        size = len(data[0]) / ratio
        size_i = int(size) + 1
        a = np.array(range(len(data[0])))
        sample_list = np.append(a[::ratio], len(data[0]) - 1)
        print(sample_list)
        for i in range(len(data)):
            o_data = data[i][sample_list]
            plt.plot(range(0, size_i), o_data, label='training accuracy')
    else:
        for i in range(len(data)):
            plt.plot(range(0, len(data[i])), data[i], label='training accuracy')

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.savefig(pic_name)
    plt.show()

def draw_plot(list,name_list, xlabel = '', ylabel = '',title = '',fz = (8,4)):
    plt.figure(figsize=fz)
    for i in range(len(list)):
        plt.plot(range(len(list[i])),list[i],label = name_list[i])
    plt.title(title)  # 标题
    plt.xlabel(xlabel)  # x轴的标签
    plt.ylabel(ylabel)  # y轴的标签
    plt.legend()
    plt.show()


def combine_bar(list, list_name, width_each = 0.1, xlabel = '', ylabel = '',title = ''):
    plt.figure()  # 建立图形
    if(len(list_name)==0):
        data = np.array(list[0]).reshape(len(list[0]))
        bar1 = plt.bar(range(len(data)), data)  # 第一个图
    else:
        for b in range(len(list)):
            data = np.array(list[b]).reshape(len(list[b]))
            name = list_name[b]
            x = range(len(data))
            num_parts = len(list)
            width_seg = width_each / num_parts
            x = [i + b * width_each / num_parts - width_each / 2 for i in x]
            bar1 = plt.bar(x, data, width=width_seg, label=name)  # 第一个图

    # plt.xticks(x, name)  # 设置x轴刻度显示值
    # plt.ylim(0, 10500)  # y轴的范围
    plt.title(title)  # 标题
    plt.xlabel(xlabel)  # x轴的标签
    plt.ylabel(ylabel)  # y轴的标签
    # plt.tight_layout()
    plt.legend()  # 设置图例
    plt.show()

#********************************************************resnet feature extractor
# a = dr.read_csv('logs/Experiments/compare_resnet_coco/result_a.csv')
# b = dr.read_csv('logs/Experiments/compare_resnet_coco/result_b.csv')
# c = dr.read_csv('logs/Experiments/compare_resnet_coco/result_c.csv')
#
# list = [np.array(a,dtype=np.float32),np.array(b,dtype=np.float32),np.array(c,dtype=np.float32)]
# list_name = ['ResNeta','ResNetb','ResNetc']
# combine_bar(list,list_name,width_each=1,'Layer Index', 'Wasserstein distance','ImageNet/COCO Decider part comparing')

# b = dr.read_csv('performance_record/wd/ImageNet_coco_decider.csv')
#
#
# list = [np.array(b,dtype=np.float32)]
# list_name = []
# combine_bar(list,list_name,width_each=1,xlabel='Layer Index', ylabel='Wasserstein distance',title='ImageNet/COCO feature extraction part comparing')
#********************************************************entropy
# a = dr.read_csv('silhouette320/train/Entropy.csv')
# b = dr.read_csv('stick320/train/Entropy.csv')
# c = dr.read_csv('WorkerData/train/Entropy.csv')
# d = dr.read_csv('silhouette_feature320/train/Entropy.csv')
# list = [np.array(a,dtype=np.float32),np.array(b,dtype=np.float32),np.array(c,dtype=np.float32),np.array(d,dtype=np.float32)]
# # print(type(a))
# print('silhouette: ',np.std(list[0][0]),np.mean(list[0][0]),np.std(list[0][1]),np.mean(list[0][1]))
# print('silhouette: ',np.std(list[1][0]),np.mean(list[1][0]),np.std(list[1][1]),np.mean(list[1][1]))
# print('silhouette: ',np.std(list[2][0]),np.mean(list[2][0]),np.std(list[2][1]),np.mean(list[2][1]))
# print('silhouette: ',np.std(list[3][0]),np.mean(list[3][0]),np.std(list[3][1]),np.mean(list[3][1]))
# # print('silhouette: ',np.std(list[0][0]),np.mean(list[0][0]),np.std(list[0][1]),np.mean(list[0][1]))
# # plt.scatter(list[0][0],list[0][1],s = 1,c = 'green')
# # plt.scatter(list[1][0],list[1][1],c = 'red')
# # plt.scatter(list[2][0],list[2][1],c = 'blue')
# plt.show()

#******************************************************performance decrease
worker_silhouette = np.array(dr.read_csv('performance_record/worker_performance_silhouette_latest.csv'),'float32')
worker_silhouette_feature = np.array(dr.read_csv('performance_record/worker_performance_silhouette_feature_latest.csv'),'float32')
worker_stick = np.array(dr.read_csv('performance_record/worker_performance_stick.csv'),'float32')
worker_stick_feature = np.array(dr.read_csv('performance_record/worker_performance_stick_feature.csv'),'float32')
worker_real = np.array(dr.read_csv('performance_record/worker_performance_real.csv'),'float32')

worker_list = []
worker_list.append(worker_real)
worker_list.append(worker_silhouette_feature)
worker_list.append(worker_silhouette)
worker_list.append(worker_stick_feature)
worker_list.append(worker_stick)

player_silhouette = np.array(dr.read_csv('performance_record/player_performance_silhouette.csv'),'float32')
player_silhouette_feature = np.array(dr.read_csv('performance_record/player_performance_silhouette_feature.csv'),'float32')
player_stick = np.array(dr.read_csv('performance_record/player_performance_stick.csv'),'float32')
player_stick_feature = np.array(dr.read_csv('performance_record/player_performance_stick_feature.csv'),'float32')
player_real = np.array(dr.read_csv('performance_record/player_performance_real.csv'),'float32')

player_list = []
player_list.append(player_real)
player_list.append(player_silhouette_feature)
player_list.append(player_silhouette)
player_list.append(player_stick_feature)
player_list.append(player_stick)

list = []
for i in range(len(worker_list)):
    list.append(worker_list[i]-player_list[i])
name_list = ['Level-0','Level-1f','Level-1','Level-2f','Level-2']
print(np.sum(list,axis=1)/150)
draw_plot(worker_list,name_list,'Epoch','mAP @ IoU = 50','Performance on Worker-Dataset')
draw_plot(player_list,name_list,'Epoch','mAP @ IoU = 50','Performance on Player-Dataset')
draw_plot(list,name_list,'Epoch','Difference','Performance Difference between Worker-Dataset and Player-Dataset')
# *******************************************************wd
# r_r = np.array(dr.read_csv('performance_record/wd/real_reallist_modify.csv'),'float32')
# s_r = np.array(dr.read_csv('performance_record/wd/silhouette_reallist_modify.csv'),'float32')
# sf_r = np.array(dr.read_csv('performance_record/wd/silhouette_feature_reallist_modify.csv'),'float32')
# st_r = np.array(dr.read_csv('performance_record/wd/stick_reallist_modify.csv'),'float32')
# stf_r = np.array(dr.read_csv('performance_record/wd/stick_feature_reallist_modify.csv'),'float32')
# #
# list = []
# list.append(r_r[:,0])
# list.append(sf_r[:,0])
# list.append(s_r[:,0])
# list.append(st_r[:,0])
# list.append(stf_r[:,0])
# list_c = []
# list_c.append(r_r[:,1])
# list_c.append(sf_r[:,1])
# list_c.append(s_r[:,1])
# list_c.append(st_r[:,1])
# list_c.append(stf_r[:,1])
# list_sum = []
# list_sum.append(np.sum(r_r,axis=1))
# list_sum.append(np.sum(sf_r,axis=1))
# list_sum.append(np.sum(s_r,axis=1))
# list_sum.append(np.sum(st_r,axis=1))
# list_sum.append(np.sum(stf_r,axis=1))
# name_list = ['Level-0','Level-1f','Level-1','Level-2','Level-2f']
# draw_plot(list,name_list,'Epoch','Wasserstein distance','Wasserstein distance changing of RPN part')
#
# draw_plot(list_c,name_list,'Epoch','Wasserstein distance','Wasserstein distance changing of Classifier part')
# draw_plot(list_sum,name_list,'Epoch','Wasserstein distance','Wasserstein distance changing')
# r_r = np.array(dr.read_csv('performance_record/wd/real_reallist_modify.csv'),'float32')
# coco_image_net = np.array(dr.read_csv('performance_record/wd/coco_imagenet_compare_modify.csv'),'float32')
# list1 = []
# list1.append(r_r[:,0])
# list1.append(coco_image_net[:,0])
# list2 = []
# list2.append(r_r[:,1])
# list2.append(coco_image_net[:,1])
# name_list = ['COCO','ImageNet']
# draw_plot(list1,name_list,'Epoch','Wasserstein distance','Wasserstein distance changing of RPN part')
# draw_plot(list2,name_list,'Epoch','Wasserstein distance','Wasserstein distance changing of Classifier part')

#******************************************************performance decrease
# worker_real_all = np.array(dr.read_csv('performance_record/worker_performance_real_all.csv'),'float32')
# worker_real = np.array(dr.read_csv('performance_record/worker_performance_real.csv'),'float32')
#
# worker_list = []
# worker_list.append(worker_real)
# worker_list.append(worker_real_all)
#
#
# player_real_all = np.array(dr.read_csv('performance_record/player_performance_real_all.csv'),'float32')
# player_real = np.array(dr.read_csv('performance_record/player_performance_real.csv'),'float32')
#
# player_list = []
# player_list.append(player_real)
# player_list.append(player_real_all)
#
# name_list = ['Block-wise training', 'All training']
# # print(np.sum(list,axis=1))
# draw_plot(worker_list,name_list,'Epoch','mAP @ IoU = 50','Performance on Worker-Dataset')
# draw_plot(player_list,name_list,'Epoch','mAP @ IoU = 50','Performance on Player-Dataset')

