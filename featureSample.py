import os

import pandas as pd
import numpy as np

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


def get_feature(path, sample_len, num, sample_type, interval=317, size=40):
    """

    :param path:  需要处理的文件名路径
    :param sample_len: 短读数据长度
    :param num: 几等分
    :param sample_type : 采样方法，1、base：连续采样 2、Windows：时间窗滑动采样 3、sample： 随机采样
    :param interval： 间隔时长，默认时长为317个点
    :param size: 总共随机采样几次，默认随机获取40个短读事件
    :return: 在当前目录生成特征文件
    """
    # 列出指定目录下的文件夹路径
    # 获取当前目录下的所有文件和子目录
    items = os.listdir(path)

    # 仅保留 CSV 文件
    file_name = [os.path.join(path, item) for item in items if os.path.isfile(os.path.join(path, item))]

    NNR_name = [item.strip('.csv') for item in items if os.path.isfile(os.path.join(path, item))]

    mol_num = len(file_name)
    # data = [pd.read_csv(i) for i in file_name]
    data = []
    for file in file_name:
        da = pd.read_csv(file)
        normal_data = da.loc[da['ProcessingStatus'] == 'normal']
        normal_data = normal_data.reset_index() # 后面选择数据需要重置后的索引
        t = normal_data.describe()
        name = file.split('\\')
        save_describe_path = os.path.join(name[0], 'describe')
        if os.path.exists(save_describe_path):
            pass
        else:
            os.mkdir(save_describe_path)
        t.to_csv(os.path.join(save_describe_path, name[1]))
        data.append(normal_data[['BlockDepth', 'ResTime']])
    # t = pd.read_excel('data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Mal-DPE-6SL-31678 events.xlsx')
    x = [i['BlockDepth'].values for i in data]
    y = [i['ResTime'].values for i in data]
    tmp_x = [m for n in x for m in n]
    tmp_y = [m for n in y for m in n]
    x_tmp = sorted(tmp_x)
    y_tmp = sorted(tmp_y)
    res_x, res_y = split_slice_n(x_tmp, y_tmp, num)
    if sample_type == 'base':
        save_path = os.path.join(path, 'featureBy' + str(num) + 'on' + str(sample_len) + 'base')
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)
        for j in range(mol_num):
            start = 0
            temp_len = data[j].shape[0]
            features = []
            for i in range(temp_len//sample_len):
                data_value = data[j][start:sample_len*(i+1)].values
                feature = get_sample(data_value, res_x, res_y, num)
                feature /= sample_len
                start = sample_len*(i+1)
                features.append(feature)
            t = pd.DataFrame(features)
            t.to_csv(os.path.join(save_path, NNR_name[j]+'-feature.csv'))

    elif sample_type == 'windows':
        save_path = os.path.join(path, 'featureBy' + str(num) + 'on' + str(sample_len) + 'windows')
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)
        for j in range(mol_num):
            start = 0
            temp_len = data[j].shape[0]
            features = []
            all_len = temp_len-sample_len
            for i in range(all_len//interval):
                data_value = data[j][start:sample_len*(i+1)].values
                feature = get_sample(data_value, res_x, res_y, num)
                feature /= sample_len
                start = i*interval
                features.append(feature)
            t = pd.DataFrame(features)
            t.to_csv(os.path.join(save_path, NNR_name[j]+'-feature.csv'))
    elif sample_type == 'random':
        save_path = os.path.join(path, 'featureBy' + str(num) + 'on' + str(sample_len) + 'random')
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)
        for index, file in enumerate(file_name):
            data_temp = data[index]
            # 假设原始纳米孔信号是一个长度为N的数组 nanopore_signal
            N = data_temp.shape[0]
            features = []
            for i in range(size):
                # 随机选择2000个索引
                selected_indices = np.random.choice(N, size=2000, replace=False)
                data_value = data_temp.loc[selected_indices].values
                feature = get_sample(data_value, res_x, res_y, num)
                feature /= sample_len
                features.append(feature)
            t = pd.DataFrame(features)
            t.to_csv(os.path.join(save_path, NNR_name[index]+'-feature.csv'))




if __name__=='__main__':
    path = './data/T240R-Gb4iGb4'
    num = 3  # 等分个数
    sample_len = 2000
    get_feature(path=path, sample_len=sample_len, num=num, sample_type='random', interval=317, size=100)