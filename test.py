import cv2
import numpy as np
from collections import Counter
from multiprocessing import Pool
import os
import dataReader
def calcEntropy(img):
    entropy = []

    hist = cv2.calcHist([img], [0], None, [256], [0, 255])
    total_pixel = img.shape[0] * img.shape[1]

    for item in hist:
        probability = item / total_pixel
        if probability == 0:
            en = 0
        else:
            en = -1 * probability * (np.log(probability) / np.log(2))
        entropy.append(en)

    sum_en = np.sum(entropy)
    return sum_en

# coding=utf-8
import cv2
import numpy as np
from collections import Counter
from multiprocessing import Pool


def calcIJ(img_patch):
    tem = img_patch.flatten()
    center_p = tem[int(len(tem) / 2)]
    mean_p = (sum(tem) - center_p) / (len(tem) - 1)
    return (center_p, mean_p)


def calcEntropy2d(img, win_w=3, win_h=3, threadNum=6):
    height = img.shape[0]
    width = img.shape[1]

    ext_x = int(win_w/2)
    ext_y = int(win_h/2)

    # 考虑滑动窗口大小，对原图进行扩边，扩展部分灰度为0
    ext_h_part = np.zeros([height, ext_x], img.dtype)
    tem_img = np.hstack((ext_h_part, img, ext_h_part))
    ext_v_part = np.zeros([ext_y, tem_img.shape[1]], img.dtype)
    final_img = np.vstack((ext_v_part, tem_img, ext_v_part))

    new_width = final_img.shape[1]
    new_height = final_img.shape[0]

    # 依次获取每个滑动窗口的内容，将其暂存在list中
    patches = []
    for i in range(ext_x, new_width - ext_x):
        for j in range(ext_y, new_height - ext_y):
            patch = final_img[j - ext_y:j + ext_y + 1, i - ext_x:i + ext_x + 1]
            patches.append(patch)

    # 并行计算每个滑动窗口对应的i、j
    # pool = Pool(processes=threadNum)
    # IJ = pool.map(calcIJ, patches)
    # pool.close()
    # pool.join()
    IJ = []
    # 串行计算速度稍慢，可以考虑使用并行计算加速
    for item in patches:
        IJ.append(calcIJ(item))

    # 循环遍历统计各二元组个数
    Fij = Counter(IJ).items()
    # 推荐使用Counter方法，如果自己写for循环效率会低很多
    # Fij = []
    # IJ_set = set(IJ)
    # for item in IJ_set:
    #     Fij.append((item, IJ.count(item)))

    # 计算各二元组出现的概率
    Pij = []
    for item in Fij:
        Pij.append(item[1] * 1.0 / (new_height * new_width))

    # 计算每个概率所对应的二维熵
    H_tem = []
    for item in Pij:
        h_tem = -item * (np.log(item) / np.log(2))
        H_tem.append(h_tem)

    # 对所有二维熵求和
    H = sum(H_tem)
    return H

def calculate_E_array(path):
    readable = ['jpg','png','bmp','JPG','BMP','PNG']
    output = []
    files = os.listdir(path)
    files.sort()
    for file in files:
        extend = os.path.splitext(file)[-1][1:]
        if extend not in readable:
            continue
        else:
            img = cv2.imread(os.path.join(path,file), cv2.IMREAD_GRAYSCALE)
            E1 = calcEntropy(img)
            E2 = calcEntropy2d(img)
            output.append([E1,E2])
            print(file, 'complete')
    dataReader.save_data(output,os.path.join(path,'Entropy.csv'))

# img1 = cv2.imread("stick320/train/B-01.jpg", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("silhouette320/train/A-1.png.JPG", cv2.IMREAD_GRAYSCALE)
# img3 = cv2.imread("WorkerData/train/1.jpg", cv2.IMREAD_GRAYSCALE)
# H1 = calcEntropy2d(img1, 3, 3)
# H2 = calcEntropy2d(img2, 3, 3)
# H3 = calcEntropy2d(img3, 3, 3)
# print(H1,H2,H3)

# calculate_E_array('stick320/train/')
# calculate_E_array('silhouette320/train/')
# calculate_E_array('WorkerData/train/')

# dirss = os.listdir('silhouette320/train/')
# dirss.sort(key= lambda x:int(x[2:]))
# for file in dirss:
#     extend = os.path.splitext(file)[-1][1:]
#     print(file,' ',extend)




# cv2.imshow("Image", img1)
# cv2.waitKey(0)