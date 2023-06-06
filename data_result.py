# -*- coding:UTF-8 -*-
# author:Lucifer_Chen
# contact: zhangchen@17888808985.com
# datetime:2023/5/30 15:14

"""
文件说明：  绘制对比图
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_comparison(file_name, sheet_name, save_name):
    """
    三种采样方式对比图
    :param file_name: 文件名
    :param sheet_name: sheet表
    :param save_name: 文照片保存名称
    :return:
    """
    # 创建一个空的DataFrame用于存储参数数据
    df = pd.DataFrame()
    # 读取第一个Excel文件的参数数据
    data1 = pd.read_excel(file_name[0], sheet_name=sheet_name)
    temp = data1.values
    df['base'] = [float(t.split('±')[0]) for t in temp[0]]
    # 读取第二个Excel文件的参数数据
    data2 = pd.read_excel(file_name[1], sheet_name=sheet_name)
    temp = data2.values
    df['random'] = [float(t.split('±')[0]) for t in temp[0]]
    # 读取第三个Excel文件的参数数据
    data3 = pd.read_excel(file_name[2], sheet_name=sheet_name)
    temp = data3.values
    df['windows'] = [float(t.split('±')[0]) for t in temp[0]]
    # 绘制柱状图
    df.plot(kind='bar')
    plt.xticks(np.arange(4), parameter_columns, rotation=15)
    plt.ylabel('Values')
    plt.title(sheet_name)
    plt.legend(['base', 'random', 'windows'])
    plt.savefig(save_name, dpi=300)
    plt.show()


def get_sample_length(file_name):

    res = []
    for file in file_name:
        data = pd.read_csv(file)
        normal_data = data.loc[data['ProcessingStatus'] == 'normal']
        res_data = normal_data[['BlockDepth', 'ResTime']]
        res_data.to_csv(file.strip('.csv')+'')
        res.append(res_data.shape[0])
    return res

if __name__ == '__main__':

    # 获取当前日期,为后面模型添加时间戳
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    # 读取Excel文件
    filename = ['./result/regressionbase2023-06-05result.xlsx', './result/regressionrandom2023-06-05result.xlsx', './result/regressionwindows2023-06-05result.xlsx']

    # 选择要比较的sheet名['LogisticRegression', 'random_forest', 'support_vector_machine', 'AdaBoost', 'naive_bayes']
    # sheetName = ['LogisticRegression', 'random_forest', 'support_vector_machine', 'AdaBoost', 'naive_bayes']
    sheetName = ['Linear',  'SVR', 'DecisionTree', 'RandomForest', 'GradientBoosting', 'NeuralNetwork', 'ExtraTree']
    condition = 's'
    # 从第一列开始，选择要比较的参数列
    # parameter_columns = ['Acc', 'precision', 'recall', 'F1']
    parameter_columns = ['R2', 'ExplainedVariance', 'MAE', 'MSE']


    if condition == 's':
        for s in sheetName:
            saveName = './picture/regressionComparison' + s + date
            get_comparison(filename, s, saveName)
            # 清空当前图形
            plt.clf()
    else:
        sheet_names = ['LogisticRegression', 'random_forest', 'support_vector_machine', 'AdaBoost', 'naive_bayes']
        dfs = [pd.read_excel(file, sheet_name=sheet_name) for sheet_name in sheet_names for file in filename]
        sample_type = 'windows'
        # 列出指定目录下的文件夹路径
        path = './data/T240R-Gb4iGb4'
        fileList = [os.path.join(path, f) for f in os.listdir(path)]
        res = get_sample_length(fileList)
        print(res, sum(res))
