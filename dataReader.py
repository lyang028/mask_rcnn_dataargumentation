import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def read_csv(path):
    csvFile = open(path, "r")
    reader = csv.reader(csvFile)
    result = []
    for item in reader:
        result.append(item)
    csvFile.close()
    return result



def get_data(path):
    Labels = read_csv(path)
    data_x = []
    address = path.split('/')
    for item in Labels:
        filename = str(int(float(item[0])))
        filepath = address[0] + '/' + filename + '.csv'
        data_x.append(read_csv(filepath))
    x_data = np.array(data_x, np.float32)

    data_y = read_csv(path)
    y_data = np.array(data_y, np.float32)
    y_data = y_data[:, 1]

    length_sample = [len(x_data[0]), len(x_data[0][0])]
    length_amount = len(x_data)
    return x_data, y_data, length_sample, length_amount


def create_difmatrix(len):
    matrix = np.eye(len - 1)
    zeroline = np.zeros([1, len - 1])
    matrix = np.r_[matrix, zeroline]

    sub_m = np.eye(len - 1)
    zeroline = np.zeros([1, len - 1])
    mm = np.r_[zeroline, sub_m]
    merge = np.array(matrix - mm, np.float32)
    return merge


def feature_extract(x_data):
    minus = np.zeros((1, 1000))
    out = np.zeros((68, 1000))
    for i in range(1000):
        minus[0][i] = i * 0.00628
    for i in range(68):
        out[i] = x_data[i] - minus
    return out


def feature_extract2(x_data):
    minus = np.zeros((1, 1000))
    out = np.zeros((68, 1000))

    for i in range(68):
        for j in range(1000):
            minus[i][j] = i * 0.00628
    for i in range(68):
        out[i] = x_data[i] - minus
    return out


def mean_extract(x_data):
    dsize1 = len(x_data)
    dsize2 = len(x_data[0])
    dsize3 = len(x_data[0][0])
    output = np.zeros((dsize1, dsize2, dsize3))
    mean_brain = np.zeros((dsize2, dsize3))

    for j in range(dsize2):
        for k in range(dsize3):
            for i in range(dsize1):
                mean_brain[j][k] += x_data[i][j][k]
            mean_brain[j][k] = mean_brain[j][k] / dsize1
    for i in range(dsize1):
        output[i] = x_data[i] - mean_brain

    return output


def mean_extract2(x_data):
    output = np.zeros((234, 68, 1000))
    mean_brain = np.zeros((68, 1000))

    for j in range(68):
        for k in range(1000):
            for i in range(234):
                mean_brain[j][k] += x_data[i][j][k]
            mean_brain[j][k] = mean_brain[j][k] / 234
    for i in range(234):
        output[i] = np.normalize_axis_index(x_data[i] - mean_brain)

    return output


def intToMatrix(iq):
    location = int(iq) - 1
    output = np.zeros((100, 1))
    output[location] = 1
    return output.T


def intToMatrix_all(y_data):
    output = np.zeros((len(y_data), 101))
    for i in range(len(y_data)):
        location = int(y_data[i])
        output[i][location] = 1
    return output


def intToK(iq):
    location = int(iq) - 1
    output = np.ones((100, 1))
    if location > 5:
        for i in range(location - 4):
            output[location - 5 - i] += 2 * i
    if location < 94:
        count = 0
        for i in range(location + 5, 100):
            output[location + 5 + count] += 2 * count
            count += 1
    return output.T


def intToK_all(y_data):
    output = np.ones((234, 101))
    for j in range(len(y_data)):
        iq = y_data[j]
        location = int(iq)
        if location > 5:
            for i in range(location - 4):
                output[j][location - 5 - i] += i
        if location < 94:
            count = 0
            for i in range(location + 5, 101):
                output[j][location + 5 + count] += count
                count += 1
    return output


def cross_validation(x, y, rate):
    len_x = len(x)
    len_train = int(len_x * rate)
    select_label = np.hstack((np.ones(len_train), np.zeros(len_x - len_train)))
    np.random.shuffle(select_label)

    x_train = x[np.array(select_label == 1)]
    x_test = x[np.array(select_label == 0)]
    y_train = y[np.array(select_label == 1)]
    y_test = y[np.array(select_label == 0)]
    return x_train, x_test, y_train, y_test, len_train, len_x - len_train


def cross_validation_reassign_copies(x, y, rate, start_label, start_label_c, number_samples_added):
    len_x = len(x)
    len_train = int(len_x * rate)
    select_label = np.hstack((np.ones(len_train), np.zeros(len_x - len_train)))
    np.random.shuffle(select_label)

    # move the copied samples to the same set as its original samples
    for i in range(number_samples_added):
        if select_label[start_label] != select_label[start_label_c]:
            select_label[start_label_c] = select_label[start_label]
            if select_label[start_label] == 1:
                len_train += 1
                print('A sample and its copy are in different sets, the copy is moved to the training set')
            else:
                len_train -= 1
                print('A sample and its copy are in different sets, the copy is moved to the testing set')
        start_label += 1
        start_label_c += 1

    # to make sure
    start_label -= 1
    start_label_c -= 1
    print('End Labels:', start_label, start_label_c)

    x_train = x[np.array(select_label == 1)]
    x_test = x[np.array(select_label == 0)]
    y_train = y[np.array(select_label == 1)]
    y_test = y[np.array(select_label == 0)]
    return x_train, x_test, y_train, y_test, len_train, len_x - len_train


def cross_validation_test(x, y, rate):
    return x[:200], x[200:], y[:200], y[200:], 200, 34


def get_minist_data(size,one_hot):
    mnist = input_data.read_data_sets("minist", one_hot=one_hot)
    if size>54999:
        size = 54999
    batch_xs, batch_ys = mnist.train.next_batch(size)
    return batch_xs, batch_ys, [28,28], size

def save_data(data, path):

    # 字典中的key值即为csv中列名
    dataframe = pd.DataFrame(data)

    # 将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(path, index=False,header=False)

def storm_data_read(path):
    csvFile = open(path, "r")
    reader = csv.reader(csvFile)
    source = []
    for item in reader:
        source.append(item)
    csvFile.close()
    output = []
    for i in range(len(source)):
        line = source[i][0]
        sline = re.split(r'\s+',line)
        print(len(sline))
        output.append(sline[1:])
    array_out = np.array(output,np.float32)
    return array_out
# def get_random_data(sample_size, length,label_range):



a = [1, 2, 3]
b = [4, 5, 6]

save_data(a,'test.csv')
# X_train, y, length_sample, length_amount = get_data("Output_source_data/Labels.csv")
# print(type(y[0]))
# X_train, y1, length_sample, length_amount = get_minist_data(1000,False)
# print(type(y1[0]))
# print(y1)