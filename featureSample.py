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
def get_data():
    pass



def get_feature(file_name, sample_len, num):
    """

    :param file_name:  需要处理的文件名列表
    :param sample_len: 短读数据长度
    :param num: 几等分
    :return: 在当前目录生成特征文件
    """
    mol_num = len(file_name)
    data = [pd.read_excel(i+'.xlsx') for i in file_name]
    t = pd.read_excel('data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Mal-DPE-6SL-31678 events.xlsx')
    x = [i['value1'].values for i in data]
    y = [i['value2'].values for i in data]
    tmp_x = [m for n in x for m in n]
    tmp_y = [m for n in y for m in n]

    x_tmp = sorted(tmp_x)
    y_tmp = sorted(tmp_y)

    res_x, res_y = split_slice_n(x_tmp, y_tmp, num)
    # print(res_x)
    # print(res_y)

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
        t.to_csv(file_name[j]+'By'+str(num)+'on'+str(sample_len)+'mol3.csv')


if __name__=='__main__':
    file_name = [
        'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Cel-DPE-6SL-28930 events',
        'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Lac-DPE-6SL-27696 events',
        'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Mal-DPE-6SL-31678 events'
    ]
    num = 3  # 等分个数
    sample_len = 2000

    get_feature(file_name, sample_len, num)