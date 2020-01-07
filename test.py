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
    return sum_en[0]

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
            output.append((E1,E2))
            print(file, 'complete')

    dataReader.save_data(output,os.path.join(path,'Entropy.csv'))

def rename_image(path,csv_name):#make the file name sortable
    list = dataReader.read_csv(os.path.join(path, csv_name))
    new_path = os.path.join(path, 'new_image')
    for i in range(len(list)-1):
        file = list[i+1][0]
        img = cv2.imread(os.path.join(path, file))
        index = int(file.split('-')[1].split('.png')[0])
        new_name = 'A-' + "%04d" % (index) + '.JPG'
        cv2.imwrite(os.path.join(new_path, new_name), img)
        list[i + 1][0] = new_name
        list[i+1][1] = os.path.getsize(os.path.join(new_path, new_name))
    dataReader.save_data(list,os.path.join(new_path, csv_name))
#
# def repair_label(path,csv_name):#change label 2 to 3 and 3 to 2
#     list = dataReader.read_csv(os.path.join(path, csv_name))
#     for i in range(len(list)-1):
#         target = list[i + 1][6].split(':')[1]
#         if(target == '"stand"}'):
#             continue
#         else if:
#
#     dataReader.save_data(list,os.path.join(new_path, csv_name))

# ***************************************************************entropy1d

# img1 = cv2.imread("figures_test_entropy/samples/l0-stand.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("figures_test_entropy/samples/l0-bend.png", cv2.IMREAD_GRAYSCALE)
# img3 = cv2.imread("figures_test_entropy/samples/l0-squat.png", cv2.IMREAD_GRAYSCALE)
#
# img11 = cv2.imread("figures_test_entropy/samples/l1-stand.png", cv2.IMREAD_GRAYSCALE)
# img21 = cv2.imread("figures_test_entropy/samples/l1-bend.png", cv2.IMREAD_GRAYSCALE)
# img31 = cv2.imread("figures_test_entropy/samples/l1-squat.png", cv2.IMREAD_GRAYSCALE)
#
# img11f = cv2.imread("figures_test_entropy/samples/l1f-stand.png", cv2.IMREAD_GRAYSCALE)
# img21f = cv2.imread("figures_test_entropy/samples/l1f-bend.png", cv2.IMREAD_GRAYSCALE)
# img31f = cv2.imread("figures_test_entropy/samples/l1f-squat.png", cv2.IMREAD_GRAYSCALE)
#
# img12 = cv2.imread("figures_test_entropy/samples/l2-stand.png", cv2.IMREAD_GRAYSCALE)
# img22 = cv2.imread("figures_test_entropy/samples/l2-bend.png", cv2.IMREAD_GRAYSCALE)
# img32 = cv2.imread("figures_test_entropy/samples/l2-squat.png", cv2.IMREAD_GRAYSCALE)
#
# img12f = cv2.imread("figures_test_entropy/samples/l2f-stand.png", cv2.IMREAD_GRAYSCALE)
# img22f = cv2.imread("figures_test_entropy/samples/l2f-bend.png", cv2.IMREAD_GRAYSCALE)
# img32f = cv2.imread("figures_test_entropy/samples/l2f-squat.png", cv2.IMREAD_GRAYSCALE)
#
# H1 = (calcEntropy(img1) + calcEntropy(img2) + calcEntropy(img3))/3
# H2 = (calcEntropy2d(img1) + calcEntropy2d(img2) + calcEntropy2d(img3))/3
#
# H11 = (calcEntropy(img11) + calcEntropy(img21) + calcEntropy(img31))/3
# H21 = (calcEntropy2d(img11) + calcEntropy2d(img21) + calcEntropy2d(img31))/3
#
# H11f = (calcEntropy(img11f) + calcEntropy(img21f) + calcEntropy(img31f))/3
# H21f = (calcEntropy2d(img11f) + calcEntropy2d(img21f) + calcEntropy2d(img31f))/3
#
# H12 = (calcEntropy(img12) + calcEntropy(img22) + calcEntropy(img32))/3
# H22 = (calcEntropy2d(img12) + calcEntropy2d(img22) + calcEntropy2d(img32))/3
#
# H12f = (calcEntropy(img12f) + calcEntropy(img22f) + calcEntropy(img32f))/3
# H22f = (calcEntropy2d(img12f) + calcEntropy2d(img22f) + calcEntropy2d(img32f))/3
#
# print("level-0(", H1,H2,')')
# print("level-1(", H11,H21,')')
# print("level-1f(", H11f,H21f,')')
# print("level-2(", H12,H22,')')
# print("level-2f(", H12f,H22f,')')

# ***************************************************************entropy2d
# img1 = cv2.imread("figures_test_entropy/samples/l1-bend.png", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("figures_test_entropy/samples/l2-bend.png", cv2.IMREAD_GRAYSCALE)
# img3 = cv2.imread("figures_test_entropy/samples/l1f-bend.png", cv2.IMREAD_GRAYSCALE)
# H1 = calcEntropy2d(img1)
# H2 = calcEntropy2d(img2)
# H11 = calcEntropy(img1)
# H21 = calcEntropy(img2)
# print('l1,l2 compare: ',H11,H1,H21,H2)
# H1 = calcEntropy2d(img1)
# H3 = calcEntropy2d(img3)
# H11 = calcEntropy(img1)
# H31 = calcEntropy(img3)
# print('l1,l2 compare: ',H11,H1,H31,H3)
#***************************************************************rename
# list = dataReader.read_csv('silhouette320_latest/train/via_export_csv.csv')
# print(list[1][0])
# fpath = os.path.join('silhouette320_latest/train',list[1][0])
# fsize = os.path.getsize(fpath)
# print(list[1][0],list[1][1], fsize)
rename_image('silhouette320_latest/val','via_export_csv.csv')
rename_image('silhouette320_latest/train','via_export_csv.csv')


