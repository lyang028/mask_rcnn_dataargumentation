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

def combine_bar(list, list_name, width_each = 0.1, xlabel = '', ylabel = '',title = ''):
    plt.figure()  # 建立图形
    for b in range(len(list)):
        data = np.array(list[b]).reshape(len(list[b]))
        name =  list_name[b]
        x = range(len(data))
        num_parts = len(list)
        width_seg = width_each/num_parts
        x = [i + b * width_each/num_parts - width_each/2 for i in x]
        bar1 = plt.bar(x, data, width= width_seg, label=name)  # 第一个图

    # plt.xticks(x, name)  # 设置x轴刻度显示值
    # plt.ylim(0, 10500)  # y轴的范围
    plt.title(title)  # 标题
    plt.xlabel(xlabel)  # x轴的标签
    plt.ylabel(ylabel)  # y轴的标签
    # plt.tight_layout()
    plt.legend()  # 设置图例
    plt.show()


a = dr.read_csv('logs/Experiments/compare_resnet_coco/result_a.csv')
b = dr.read_csv('logs/Experiments/compare_resnet_coco/result_b.csv')
c = dr.read_csv('logs/Experiments/compare_resnet_coco/result_c.csv')

list = [np.array(a,dtype=np.float32),np.array(b,dtype=np.float32),np.array(c,dtype=np.float32)]
list_name = ['ResNeta','ResNetb','ResNetc']
combine_bar(list,list_name,width_each=1)