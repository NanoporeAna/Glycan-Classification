# -*- coding:UTF-8 -*-
# author:Lucifer_Chen
# contact: zhnagchen@17888808985.com
# datetime:2023/5/29 10:02

"""
文件说明：  
"""
import os
import time
from datetime import datetime

import openpyxl
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

def plot_p(x, y, save_name):
    # 统计每个数据的个数
    data_counts = {}
    offset = 0.05  # 调整标记的位置偏移量
    for i in range(len(x)):
        data = (x[i], y[i])
        if data in data_counts:
            data_counts[data] += 1
        else:
            data_counts[data] = 1

    # 提取数据和计数
    data = list(data_counts.keys())

    counts = list(data_counts.values())
    # 提取计数的最大值和最小值，用于归一化
    max_count = max(counts)
    min_count = min(counts)

    # 绘制散点图
    plt.scatter([d[0] for d in data], [d[1] for d in data], c=counts, marker='.', s=8)

    # # 标记数据个数
    # for i in range(len(data)):
    #     count = counts[i]
    #     plt.annotate(str(count), (data[i][0], data[i][1] + offset), color='red')
    # 绘制对角线
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    # 设置图形属性
    plt.xlabel('Test')
    plt.ylabel('Predict')
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    plt.grid(True)
    plt.savefig(save_name, dpi=300)
    # 显示图形
    plt.show()

def load_regression_forest_data(file_name, label):
    """
    加载样本数据，分层抽样
    :param file_name: 文件列表
    :param label 上面文件对应的标签
    :return:
    """
    features = []
    labels = []
    for index, file in enumerate(file_name):
        raw_data = pd.read_csv(file, header=0)  # 读取csv数据，并将第一行视为表头，返回DataFrame类型
        length = raw_data.shape[0]
        data = raw_data.values
        features.extend(data[::, 1::])
        labels.extend([label[index]] * length)
    # 选取20%数据作为测试集，剩余为训练集
    # stratify=labels 这个是分组的标志，用到自己数据集上即为四个分子的类别，保证每个分子取到的样本数差不多，分层抽样
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.2, stratify=labels)
    return train_features, test_features, train_labels, test_labels


def train(file_name, label, model_type, sample_type, nmr_name):
    # 获取当前日期,为后面模型添加时间戳
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    # 结果保存
    save_path = '../result/regression' + sample_type + date + 'result.xlsx'
    if not os.path.exists(save_path):
        df = pd.DataFrame()  # 表示创建空的表
        df.to_excel(save_path)
    mol_num = len(file_name)
    time_2 = time.time()
    all_test_labels = []
    all_test_pres = []
    test_acc = []
    train_acc = []
    all_confusion_matrix = []
    confusion_matrix = np.zeros((mol_num, mol_num))
    confusion_matrix.tolist()
    # 各项指标
    all_evs, all_mae, all_mse, all_r2s, all_roc = [], [], [], [], []

    # 创建模型
    if model_type == 'Linear':
        model = LinearRegression()
    # elif model_type == 'Polynomial':
    #     model = PolynomialFeatures(degree=2)
    elif model_type == 'SVR':
        model = SVR(kernel='rbf')
    elif model_type == 'DecisionTree':
        model = DecisionTreeRegressor()
    elif model_type == 'RandomForest':
        model = RandomForestRegressor(n_estimators=80)
    elif model_type == 'GradientBoosting':
        model = GradientBoostingRegressor(n_estimators=100)
    elif model_type == 'NeuralNetwork':
        model = MLPRegressor()
    elif model_type == 'ExtraTree':
        model = ExtraTreeRegressor()
    else:
        raise ValueError("Invalid condition specified")
    epoch = 100

    for k in range(10):
        evs, mae, mse, r2s, roc = [], [], [], [], []
        for i in range(epoch):
            print('prepare datasets...(every time)')
            # 自己数据集加载
            train_features, test_features, train_labels, test_labels = load_regression_forest_data(file_name, label)
            print('Start training...')

            model.fit(train_features, train_labels)  # training the model
            train_score = model.score(train_features, train_labels)
            train_acc.append(train_score)
            print("训练集：", train_score)

            joblib.dump(model, "../models/regressor" + model_type + date + sample_type + '.m')
            time_3 = time.time()
            print('training cost %f seconds' % (time_3 - time_2))

            print('Start predicting...')
            test_predict = model.predict(test_features)
            all_test_labels.extend(test_labels)
            all_test_pres.extend(test_predict)
            # 获取验证集的准确率
            test_score = model.score(test_features, test_labels)
            test_acc.append(test_score)
            print("The test accruacy score is %f" % test_score)
            time_4 = time.time()
            print('predicting cost %f seconds' % (time_4 - time_3))

            # 采用混淆矩阵（metrics）计算各种评价指标
            evs.append(metrics.explained_variance_score(test_labels, test_predict))
            mae.append(metrics.mean_absolute_error(test_labels, test_predict))
            mse.append(metrics.mean_squared_error(test_labels, test_predict))
            r2s.append(metrics.r2_score(test_labels, test_predict))

        print('*' * 10, 'one hundray time', '*' * 10)
        print('解释方差：%.5f' % (sum(evs) / 100))
        all_evs.append(sum(evs) / 100)
        print('平均绝对值误差：%.5f' % (sum(mae) / 100))
        all_mae.append(sum(mae) / 100)
        print('均方误差: %.5f' % (sum(mse) / 100))
        all_mse.append(sum(mse) / 100)
        print("可决系数:%.5f" % (sum(r2s) / 100))
        all_r2s.append(sum(r2s) / 100)

    print('最终结果')
    # all_acc = np.mean(test_acc)
    # all_acc_std = np.std(test_acc)
    # print('测试精度： %.5f' % all_acc)
    # print(all_acc_std)
    evs_mean = np.mean(all_evs)
    evs_std = np.std(all_evs, ddof=1)
    print('解释方差：%.5f' % evs_mean)
    print(evs_std)
    mae_mean = np.mean(all_mae)
    mae_std = np.std(all_mae, ddof=1)
    print('平均绝对值误差：%.5f' % mae_mean)
    print(mae_std)
    mse_mean = np.mean(all_mse)
    mse_std = np.std(all_mse, ddof=1)
    print('均方误差：%.5f' % mse_mean)
    print(mse_std)
    r2s_mean = np.mean(all_r2s)
    r2s_std = np.std(all_r2s, ddof=1)
    print('可决系数：%.5f' % r2s_mean)
    print(r2s_std)

    saveName = '../picture/regression' + model_type + sample_type + date
    pictures_dataframe_path = '../result/regression' + model_type + sample_type + date + 'forecast.csv'
    plot_p(x=all_test_labels, y=all_test_pres, save_name=saveName)
    pictures_dataframe = {'True label': all_test_labels, 'Forecast': all_test_pres}
    pictures_dataframe = pd.DataFrame(pictures_dataframe)
    pictures_dataframe.to_csv(pictures_dataframe_path)

    data = {'R2': str(round(r2s_mean, 4)) + '±' + str(round(r2s_std, 4)),
            'ExplainedVariance': str(round(evs_mean, 4)) + '±' + str(round(evs_std, 4)),
            'MAE': str(round(mae_mean, 4)) + '±' + str(round(mae_std, 4)),
            'MSE': str(round(mse_mean, 4)) + '±' + str(round(mse_std, 4))}

    with pd.ExcelWriter(save_path, mode='a', engine='openpyxl') as writer:
        df = pd.DataFrame(data, index=[0])
        df.to_excel(writer, sheet_name=model_type, index=False)
        df1 = pd.DataFrame(confusion_matrix)
        df1.to_excel(writer, sheet_name=model_type + '_confusion_matrix', header=False, index=False)

    # 删除空表头Sheet1
    # 执行删除操作：
    sheet_name = 'Sheet1'
    # 载入工作簿
    try:
        workbook = openpyxl.load_workbook(save_path)
        # 删除目标Sheet
        worksheet = workbook[sheet_name]
        workbook.remove(worksheet)
        # 保存已做删除处理的工作簿
        workbook.save(save_path)
    except:
        print('No sheet1')
    time_5 = time.time()
    print('总共花费时长： %f second' % (time_5-time_2))


if __name__ == '__main__':
    conditions = ['Linear',  'SVR', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'NeuralNetwork', 'ExtraTree']
    sampleTypes = ['random', 'windows', 'base']
    path = '../data/T240R-Gb4iGb4/featureBy3on2000'
    # conditions, sampleTypes = [], None
    if isinstance(conditions, list) and isinstance(sampleTypes, list):
        for sampleType in sampleTypes:
            tt = path + sampleType
            for condition in conditions:
                # 列出指定目录下的文件夹路径

                fileName = [os.path.join(tt, f) for f in os.listdir(tt)]
                nmrName = [f.strip('-feature.csv') for f in os.listdir(tt)]  # 使用的数据名称
                train(file_name=fileName, label=[0.2, 0.35, 0.5, 0.65, 0.8, 1, 0], model_type=condition, sample_type=sampleType, nmr_name=nmrName)
    else:
        condition = 'RandomForest'  # 算法开关 ['Linear', 'Polynomial', 'SVR', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'NeuralNetwork', 'ExtraTree']
        sampleType = 'random'
        path = path + sampleType  # 列出指定目录下的文件夹路径
        fileName = [os.path.join(path, f) for f in os.listdir(path)]
        nmrName = [f.split('eventMD')[0] for f in os.listdir(path)]  # 使用的数据名称
        train(file_name=fileName, label=[0.2, 0.35, 0.5, 0.65, 0.8, 1, 0], model_type=condition, sample_type=sampleType, nmr_name=nmrName)
