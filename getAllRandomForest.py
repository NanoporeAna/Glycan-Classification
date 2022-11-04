import time

import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


def split_slice_n(x, y, n):
    length = len(x)
    length = length//n
    res_x, res_y = [], []
    for i in range(1, n):
        res_x.append(x[length*i])
        res_y.append(y[length*i])
    return res_x, res_y

def get_index(data, x, y, length):
    """
    根据输入的一个点判断属于我们划分区域的哪一块
    :param data: 输入一个点，二维数据
    :param x: x轴阈值列表
    :param y: y轴阈值列表
    :param length: 阈值几等分
    :return: 属于哪份区域
    """
    index_x, index_y = -2, -2
    for index, i in enumerate(x):
        if data[0] < i:
            index_x = index
            break
    for index, i in enumerate(y):
        if data[1] < i:
            index_y = index
            break
    if index_x == -2:
        index_x = length-1
    if index_y == -2:
        index_y = length-1
    return index_y + index_x * length

def get_sample(data, res_x, res_y, num):
    feature = np.zeros(num * num)
    for d in data:
        inx = get_index(d, res_x, res_y, num)
        feature[inx] += 1
    return feature
def get_data():
    pass

file_name = ['data/Cel-DPE6SL-28930 events', 'data/Lac-DPE6SL-27696 events',
             'data/Mal-DPE6SL-31678 events']
def getdata(sample_len, num):
    data = [pd.read_excel(i+'.xlsx') for i in file_name]
    x = [i['value1'].values for i in data]
    y = [i['value2'].values for i in data]
    tmp_x = [m for n in x for m in n]
    tmp_y = [m for n in y for m in n]

    x_tmp = sorted(tmp_x)
    y_tmp = sorted(tmp_y)
    res_x, res_y = split_slice_n(x_tmp, y_tmp, num)
    features = []
    labels = []
    for j in range(3):
        start = 0
        temp_len = data[j].shape[0]
        for i in range(temp_len//sample_len):
            data_value = data[j][start:sample_len*(i+1)].values
            feature = get_sample(data_value, res_x, res_y, num)
            feature /= sample_len
            start = sample_len*(i+1)
            features.append(feature)
            labels.append(j)

    return features, labels

if __name__ == '__main__':
    epoch = 100
    sample_len = 1000
    split_num = 4
    print('prepare datasets...')
    # Iris数据集
    # iris=datasets.load_iris()
    # features=iris.data
    # labels=iris.target
    time_2 = time.time()
    model_path = None


    for length in range(1000, 3001, 100):
        res_test, res_train = [], []

        for split_slice_num in range(3, 51):
            x = np.arange(3, 51)
            print('样本短读时间为%d' % length)
            print('在%d等分下：' % split_slice_num)
            test_acc = []
            train_acc = []
            fea, lab = getdata(length, split_slice_num)
            for i in range(epoch):
                # 自己数据集加载
                train_features, test_features, train_labels, test_labels = train_test_split(fea, lab,
                                                                                            test_size=0.2, stratify=lab)

                # n_estimators表示要组合的弱分类器个数；
                # algorithm可选{‘SAMME’, ‘SAMME.R’}，默认为‘SAMME.R’，表示使用的是real boosting算法，‘SAMME’表示使用的是discrete boosting算法
                clf = RandomForestClassifier(n_estimators=25)
                clf.fit(train_features, train_labels)  # training the svc model
                train_score = clf.score(train_features, train_labels)
                train_acc.append(train_score)

                joblib.dump(clf, "randomForest_model.m")
                time_3 = time.time()


                test_predict = clf.predict(test_features)
                # 获取验证集的准确率
                test_score = clf.score(test_features, test_labels)
                test_acc.append(test_score)
                time_4 = time.time()
            mean_acc_train = sum(train_acc) / 100
            mean_acc_test = sum(test_acc) / 100
            res_test.append(mean_acc_test)
            res_train.append(mean_acc_train)
        plt.plot(x, res_train, c="green", label=r'train')
        plt.plot(x, res_test,  c="red", label=r'test')
        plt.title('randomForest')
        plt.legend()
        plt.savefig('./picture/randomForest'+str(length)+'.png')
        fig = plt.gcf()  # 获取当前figure
        plt.close(fig)  # 关闭传入的 figure 对象

