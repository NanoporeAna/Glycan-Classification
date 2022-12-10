import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_name = ['data/3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB1/1-3SG-MPB---40000 events.xlsx',
             'data/3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB1/2-3SL-MPB---40000 events.xlsx',
             'data/3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB1/3-STetra2-MPB---40000 events.xlsx',
             'data/3SG-MPB---3SL-MPB---STetra2-MPB---LSTa-MPB1/4-LSTa-MPB---40000 events.xlsx']
data = [pd.read_excel(i) for i in file_name]
plt.yscale('symlog')
plt.ylim([0, 100])
name = 'ALLSample'
plt.scatter(data[0]['value1'][:].values, data[0]['value2'][:].values, s=2, marker='.', c='r')
plt.scatter(data[1]['value1'][:].values, data[1]['value2'][:].values, s=2, marker='.', c='g')
plt.scatter(data[2]['value1'][:].values, data[2]['value2'][:].values, s=2, marker='.', c='y')
plt.scatter(data[3]['value1'][:].values, data[3]['value2'][:].values, s=2, marker='.', c='b')
# 画出 x=2 这条垂直线
plt.axvline(0.634429157295902, linestyle="--")
plt.axvline(0.711002490116433, linestyle="--")

# 画出 y=1 这条水平线
plt.axhline(0.265852989798345, linestyle="--")
plt.axhline(0.324150593740613, linestyle="--")

plt.xlabel('Ib/I0')
plt.ylabel('Dwell time(ms)')
# plt.title(name)
plt.savefig(name+'.svg')
plt.savefig(name+'.png')

