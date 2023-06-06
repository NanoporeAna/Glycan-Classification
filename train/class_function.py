# -*- coding:UTF-8 -*-
# author:Lucifer_Chen
# contact: zhnagchen@17888808985.com
# datetime:2023/5/12 14:27
"""
文件说明：
"""
import os
import time
from datetime import datetime
import joblib
import numpy as np
import openpyxl
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def polt_conf(cm, classes, name):
    """

    :param cm: 混淆矩阵
    :param classes: 类别名称
    :param name: 保存图像名
    :return:
    """
    # 绘制混淆矩阵图像
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes, ylabel='True label', xlabel='Predicted label')
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(name, dpi=300)
    plt.show()


def load_forest_data(file_name):
    """
    加载样本数据，分层抽样
    :param file_name:
    :return:
    """
    features = []
    labels = []
    for index, file in enumerate(file_name):
        raw_data = pd.read_csv(file, header=0)  # 读取csv数据，并将第一行视为表头，返回DataFrame类型
        length = raw_data.shape[0]
        data = raw_data.values
        features.extend(data[::, 1::])
        labels.extend([index] * length)
    # 选取20%数据作为测试集，剩余为训练集
    # stratify=labels 这个是分组的标志，用到自己数据集上即为四个分子的类别，保证每个分子取到的样本数差不多，分层抽样
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.2, stratify=labels)
    return train_features, test_features, train_labels, test_labels


def train(file_name, model_type, sample_type, nmr_name):
    # 获取当前日期,为后面模型添加时间戳
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    # 结果保存
    save_path = '../result/' + sample_type + date + 'result.xlsx'
    if not os.path.exists(save_path):
        df = pd.DataFrame()  # 表示创建空的表
        df.to_excel(save_path)
    mol_num = len(file_name)
    time_2 = time.time()
    test_acc = []
    train_acc = []
    all_confusion_matrix = []
    confusion_matrix = np.zeros((mol_num, mol_num))
    confusion_matrix.tolist()
    # 各项指标
    all_ps, all_rs, all_fs, all_cs, all_roc = [], [], [], [], []
    # 根据条件选择要使用的机器学习算法
    if model_type == "naive_bayes":
        model = MultinomialNB(alpha=1.0)  # 加入laplace平滑

    elif model_type == "AdaBoost":
        model = AdaBoostClassifier(n_estimators=100, algorithm='SAMME.R')
    elif model_type == "support_vector_machine":
        """
           C：SVC的惩罚参数，默认值是1.0；C越大，对误分类的惩罚增大，趋向于对训练集完全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。
           C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
           kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
        """
        model = svm.SVC(C=1.0, kernel='rbf')  # poly效果最好
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=80)  # 容易区分则设置为25
    elif model_type == "LogisticRegression":
        model = linear_model.LogisticRegression()
    else:
        raise ValueError("Invalid condition specified")
    epoch = 100

    for k in range(10):
        ps, rs, fs, cs, roc = [], [], [], [], []
        for i in range(epoch):
            print('prepare datasets...(every time)')
            # 自己数据集加载
            train_features, test_features, train_labels, test_labels = load_forest_data(file_name)
            print('Start training...')

            model.fit(train_features, train_labels)  # training the model
            train_score = model.score(train_features, train_labels)
            train_acc.append(train_score)
            print("训练集：", train_score)

            joblib.dump(model, "../models/" + model_type + date + sample_type + '.m')
            time_3 = time.time()
            print('training cost %f seconds' % (time_3 - time_2))

            print('Start predicting...')
            test_predict = model.predict(test_features)
            # 获取验证集的准确率
            test_score = model.score(test_features, test_labels)
            test_acc.append(test_score)
            print("The test accruacy score is %f" % test_score)
            time_4 = time.time()
            print('predicting cost %f seconds' % (time_4 - time_3))

            # 采用混淆矩阵（metrics）计算各种评价指标
            ps.append(metrics.precision_score(test_labels, test_predict, average='weighted'))
            rs.append(metrics.recall_score(test_labels, test_predict, labels=np.arange(mol_num), average='weighted'))
            fs.append(metrics.f1_score(test_labels, test_predict, average='weighted'))
            cs.append(np.mean(test_labels == test_predict))

            # 分类报告
            class_report = metrics.classification_report(test_labels, test_predict,
                                                         target_names=nmr_name)
            print(class_report)
            # 输出混淆矩阵
            confusion_matrix += metrics.confusion_matrix(test_labels, test_predict)

        print('--混淆矩阵--')
        print(confusion_matrix)
        all_confusion_matrix.append(confusion_matrix)
        print('*' * 10, 'one hundray time', '*' * 10)
        print('精准值：%.5f' % (sum(ps) / 100))
        all_ps.append(sum(ps) / 100)
        print('召回率：%.5f' % (sum(rs) / 100))
        all_rs.append(sum(rs) / 100)
        print('F1: %.5f' % (sum(fs) / 100))
        all_fs.append(sum(fs) / 100)
        print("准确率:%.5f" % (sum(cs) / 100))
        all_cs.append(sum(cs) / 100)
    print('最终结果')
    ps_mean = np.mean(all_ps)
    ps_std = np.std(all_ps, ddof=1)
    print('精准值：%.5f' % ps_mean)
    print(ps_std)
    rs_mean = np.mean(all_rs)
    rs_std = np.std(all_rs, ddof=1)
    print('召回率：%.5f' % rs_mean)
    print(rs_std)
    fs_mean = np.mean(all_fs)
    fs_std = np.std(all_fs, ddof=1)
    print('F1：%.5f' % fs_mean)
    print(fs_std)
    cs_mean = np.mean(all_cs)
    cs_std = np.std(all_cs, ddof=1)
    print('准确率：%.5f' % cs_mean)
    print(cs_std)
    print('混淆矩阵*')
    print(confusion_matrix)
    polt_conf(confusion_matrix, nmr_name, "../picture/" + model_type + date + sample_type + '.png')
    data = {'Acc': str(round(cs_mean, 4)) + '±' + str(round(cs_std, 4)),
            'precision': str(round(ps_mean, 4)) + '±' + str(round(ps_std, 4)),
            'recall': str(round(rs_mean, 4)) + '±' + str(round(rs_std, 4)),
            'F1': str(round(fs_mean, 4)) + '±' + str(round(fs_std, 4))}

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


if __name__ == '__main__':
    # conditions = ['LogisticRegression', 'random_forest', 'support_vector_machine', 'AdaBoost', 'naive_bayes']
    # sampleTypes = ['windows', 'base']
    conditions, sampleTypes = False, False
    if conditions and sampleTypes:
        for sampleType in sampleTypes:
            for condition in conditions:
                # 列出指定目录下的文件夹路径
                path = '../data/T240R二糖-NNR/featureBy3on2000' + sampleType
                fileName = [os.path.join(path, f) for f in os.listdir(path)]
                nmrName = [f.strip('-feature.csv') for f in os.listdir(path)]  # 使用的数据名称
                train(file_name=fileName, model_type=condition, sample_type=sampleType, nmr_name=nmrName)
    else:
        condition = 'random_forest'  # 算法开关
        sampleType = 'random'
        # 列出指定目录下的文件夹路径
        path = '../data/T240R二糖-NNR/featureBy3on2000' + sampleType
        fileName = [os.path.join(path, f) for f in os.listdir(path)]
        nmrName = [f.strip('-feature.csv') for f in os.listdir(path)]  # 使用的数据名称
        train(file_name=fileName, model_type=condition, sample_type=sampleType, nmr_name=nmrName)

