import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


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
    # t = pd.read_excel('data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Mal-DPE-6SL-31678 events.xlsx')
    x = [i['value1'].values for i in data]
    y = [i['value2'].values for i in data]
    tmp_x = [m for n in x for m in n]
    tmp_y = [m for n in y for m in n]

    x_tmp = sorted(tmp_x)
    y_tmp = sorted(tmp_y)

    res_x, res_y = split_slice_n(x_tmp, y_tmp, num)
    df = {'Ib/IO': res_x, 'Dwell time': res_y}
    df = pd.DataFrame(df)
    df.to_csv('split_slice1212.csv')
    # print(res_x)
    # print(res_y)

    feature_sa = []
    # 获取单个分子全局分布的样本特征
    for i in range(mol_num):
        data_value = data[i][:].values
        all_sample_len = data[i].shape[0]
        feature = get_sample(data_value, res_x, res_y, num)
        feature /= all_sample_len
        feature_sa.append(feature)
    print(feature_sa)
    t = pd.DataFrame(feature_sa)
    sns.set()
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示，必须放在sns.set之后
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(t.values, ax=ax, vmin=0, vmax=0.3, cmap='YlOrRd', annot=True, linewidths=1, cbar=True)
    # ax.set_title('四分子热力图')  # plt.title('热图'),均可设置图片标题
    ax.set_ylabel('molecule')  # 设置纵轴标签
    ax.set_xlabel('feature')  # 设置横轴标签
    # 设置坐标字体方向，通过rotation参数可以调节旋转角度
    x = [0.5, 1.5, 2.5, 3.5]
    plt.yticks(x, ['3SG', '3SL', 'LSTa', 'STetra2'])
    label_y = ax.get_yticklabels()
    plt.setp(label_y, rotation=360, horizontalalignment='right')
    label_x = ax.get_xticklabels()
    plt.setp(label_x, rotation=45, horizontalalignment='right')
    plt.savefig('feature-2022-12-11-2.svg')
    plt.show()


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
    # file_name = [
    #     'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Cel-DPE-6SL-28930 events',
    #     'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Lac-DPE-6SL-27696 events',
    #     'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Mal-DPE-6SL-31678 events'
    # ]
    file_name = ['data/3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB1/1-3SG-MPB---40000 events',
                 'data/3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB1/2-3SL-MPB---40000 events',
                 'data/3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB1/4-LSTa-MPB---40000 events',
                 'data/3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB1/3-STetra2-MPB---40000 events']
    num = 3  # 等分个数
    sample_len = 2000

    get_feature(file_name, sample_len, num)