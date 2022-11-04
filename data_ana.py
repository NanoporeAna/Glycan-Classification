import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns
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
file_name = [
            'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Cel-DPE-6SL-28930 events',
             'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Lac-DPE-6SL-27696 events',
             'data/Lac-DPE-6SL---Cel-DPE-6SL---Mal-DPE-6SL/Mal-DPE-6SL-31678 events'
             ]
mol_num = len(file_name)
data = [pd.read_excel(i+'.xlsx') for i in file_name]
x = [i['value1'].values for i in data]
y = [i['value2'].values for i in data]
tmp_x = [m for n in x for m in n]
tmp_y = [m for n in y for m in n]

x_tmp = sorted(tmp_x)
y_tmp = sorted(tmp_y)
num = 3  # 几等分
res_x, res_y = split_slice_n(x_tmp, y_tmp, num)
feature_sa = []
# 获取单个分子全局分布的样本特征
for i in range(mol_num):
    data_value = data[i][:].values
    sample_len = data[i].shape[0]
    feature = get_sample(data_value, res_x, res_y, num)
    feature /= sample_len
    feature_sa.append(feature)
print(feature_sa)
t = pd.DataFrame(feature_sa)
sns.set()
plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示，必须放在sns.set之后
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(t.values, ax=ax, vmin=0, vmax=0.3, cmap='YlOrRd', annot=True, linewidths=1, cbar=True)
ax.set_title('四分子热力图')  # plt.title('热图'),均可设置图片标题
ax.set_ylabel('molecule')   # 设置纵轴标签
ax.set_xlabel('feature')   # 设置横轴标签
# 设置坐标字体方向，通过rotation参数可以调节旋转角度
x = [0.5, 1.5, 2.5, 3.5]
plt.yticks(x, ['3SGP', '3SLP', 'LSTaP', 'STetraP'])
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
label_x = ax.get_xticklabels()
plt.setp(label_x, rotation=45, horizontalalignment='right')
plt.savefig('allfig.svg')
plt.show()

start, end = 900, 15000
# 计算四个分子的KL散度
all_kls = []
for j in range(mol_num):
    kls = []
    for sample_len in range(start, end):
        flag = 0
        temp_len = data[j].shape[0]
        features = []
        KL = 0
        nn = 0
        for i in range(temp_len//sample_len):
            data_value = data[j][flag:sample_len * (i + 1)].values
            feature = get_sample(data_value, res_x, res_y, num)
            feature /= sample_len
            flag = sample_len * (i + 1)
            # 计算单个样本的KL散度
            KL += scipy.stats.entropy(feature_sa[j], feature)
            nn += 1
        KL /= nn
        kls.append(KL)
    all_kls.append(kls)

x = np.arange(start, end)
x = x.tolist()
plt.plot(x, all_kls[0], color='orangered',)
plt.plot(x, all_kls[1], color='blueviolet',)
plt.plot(x, all_kls[2], color='green', )
plt.xlabel('number of sample')
plt.ylabel('KL')
plt.title('mol split by' + str(num))
plt.savefig('KL.svg')
plt.show()