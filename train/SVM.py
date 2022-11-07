import os
import time
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import auc, roc_curve, ConfusionMatrixDisplay
from sklearn import datasets, metrics
from sklearn import svm
from sklearn.model_selection import train_test_split

file_name = ['../data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Cel-DPE-6SL-28930 events',
             '../data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Lac-DPE-6SL-27696 events',
             '../data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Mal-DPE-6SL-31678 events']
mol_num = len(file_name)
for i in range(mol_num):
    file_name[i] += 'By3on2000mol3.csv'

def load_svm_data(file_name):
    features = []
    labels = []
    for index, file in enumerate(file_name):
        raw_data = pd.read_csv(file, header=0)  # 读取csv数据，并将第一行视为表头，返回DataFrame类型
        length = raw_data.shape[0]
        data = raw_data.values
        features.extend(data[::, 1::])
        labels.extend([index]*length)
     # 选取20%数据作为测试集，剩余为训练集
    # stratify=labels 这个是分组的标志，用到自己数据集上即为四个分子的类别，保证每个分子取到的样本数差不多，分层抽样
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels,
                                                                                test_size=0.2, stratify=labels)
    return train_features, test_features, train_labels, test_labels

if __name__ == '__main__':
    epoch = 100
    print('prepare datasets...')
    # 自己数据集加载
    save_path = '3ML_result.xlsx'
    if not os.path.exists(save_path):
        df = pd.DataFrame()  # 表示创建空的表
        df.to_excel(save_path)
    time_2 = time.time()
    model_path = None
    test_acc = []
    train_acc = []
    all_confusion_matrix = []
    confusion_matrix = np.zeros((3, 3))
    confusion_matrix.tolist()
    all_ps, all_rs, all_fs, all_cs, all_roc = [], [], [], [], []
    for k in range(10):
        ps, rs, fs, cs, roc = [], [], [], [], []
        for i in range(epoch):
        # if model_path is None:
            # 自己数据集加载
            train_features, test_features, train_labels, test_labels = load_svm_data(file_name)
            print('Start training...')
            """
            C：SVC的惩罚参数，默认值是1.0；C越大，对误分类的惩罚增大，趋向于对训练集完全分对的情况，这样对训练集测试时准确率很高，但泛化能力弱。
            C值小，对误分类的惩罚减小，允许容错，将他们当成噪声点，泛化能力较强。
            kernel ：核函数，默认是rbf，可以是‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
            """
            clf = svm.SVC(C=1.0, kernel='RBF')  # poly效果最好
            clf.fit(train_features, train_labels)  # training the svc train
            train_score = clf.score(train_features, train_labels)
            train_acc.append(train_score)
            print("训练集：", train_score)

            joblib.dump(clf, "../models/svm_model%d.m" % i)
            time_3 = time.time()
            print('training cost %f seconds' % (time_3 - time_2))

            print('Start predicting...')
            test_predict = clf.predict(test_features)
            # 获取验证集的准确率
            # test_score = clf.score(test_labels, test_predict)
            test_score = clf.score(test_features, test_labels)
            test_acc.append(test_score)

            print("The test accruacy score is %f" % test_score)
            time_4 = time.time()
            print('predicting cost %f seconds' % (time_4 - time_3))

            # 采用混淆矩阵（metrics）计算各种评价指标
            ps.append(metrics.precision_score(test_labels, test_predict, average='weighted'))
            rs.append(metrics.recall_score(test_labels, test_predict, labels=[0,1,2], average='weighted'))
            fs.append(metrics.f1_score(test_labels, test_predict, average='weighted'))
            cs.append(np.mean(test_labels == test_predict))
            # f, t, _ = roc_curve(test_labels, test_predict)
            # roc.append(auc(f, t))
            # 分类报告
            class_report = metrics.classification_report(test_labels, test_predict,
                                                         target_names=[ "Cel-DPE6SL",
                                                                       "Lac-DPE6SL", "Mal-DPE6SL"])
            print(class_report)
            """
            test_labels 为样本真实标签，test_predict 为样本预测标签；
            support：当前行的类别在测试数据中的样本总量，如上表就是，在class 0 类别在测试集中总数量为1；
            precision：精度=正确预测的个数(TP)/被预测正确的个数(TP+FP)；人话也就是模型预测的结果中有多少是预测正确的
            recall:召回率=正确预测的个数(TP)/预测个数(TP+FN)；人话也就是某个类别测试集中的总量，有多少样本预测正确了；
            f1-score:F1 = 2*精度*召回率/(精度+召回率)
            micro avg：计算所有数据下的指标值，假设全部数据 5 个样本中有 3 个预测正确，所以 micro avg 为 3/5=0.6
            macro avg：每个类别评估指标未加权的平均值，比如准确率的 macro avg，(0.50+0.00+1.00)/3=0.5
            weighted avg：加权平均，就是测试集中样本量大的
            """
            # 输出混淆矩阵
            # confusion_matrix = metrics.confusion_matrix(test_labels, test_predict)
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
    data = {'Acc': str(round(cs_mean, 4)) + '±' + str(round(cs_std, 4)),
            'precision': str(round(ps_mean, 4)) + '±' + str(round(ps_std, 4)),
            'recall': str(round(rs_mean, 4)) + '±' + str(round(rs_std, 4)),
            'F1': str(round(fs_mean, 4)) + '±' + str(round(fs_std, 4))}
    with pd.ExcelWriter(save_path, mode='a', engine='openpyxl') as writer:
        df = pd.DataFrame(data, index=[0])

        df.to_excel(writer, sheet_name='SVM', index=False)
        df1 = pd.DataFrame(confusion_matrix)
        df1.to_excel(writer, sheet_name='svm_confusion_matrix', header=False, index=False)

    #         # 输出混淆矩阵
    #         confusion_matrix += metrics.confusion_matrix(test_labels, test_predict)
    # print('--混淆矩阵--')
    # print(confusion_matrix)
    # # 画出混淆矩阵
    # # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    # labels = ['Cel', 'Lac', 'Mal']
    # disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    # disp.plot()
    # plt.title('SVM')
    # plt.savefig('../picture/3 modelSVM best.png')
    # plt.show()
    # mean_acc_train = sum(train_acc)/100
    # mean_acc_test = sum(test_acc)/100
    # print("The mean of train accruacy score is %f" % mean_acc_train)
    # print("The mean of test accruacy score is %f" % mean_acc_test)
    # print('*'*10, 'one hundray time', '*'*10)
    # print('精准值：%.5f' % (sum(ps)/100))
    # print('召回率：%.5f' % (sum(rs)/100))
    # print('F1: %.5f' % (sum(fs)/100))
    # print("准确率:%.5f" % (sum(cs)/100))
    # # print("roc:%.5f" % (sum(roc) / 100))

