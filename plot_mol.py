import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
file_name = ['data/3SGP---63554 events.xlsx', 'data/3SLP---62910 events.xlsx',
        'data/LSTaP---64097 events.xlsx', 'data/STetraP---63445 events.xlsx']
data = [pd.read_excel(i) for i in file_name]
plt.yscale('symlog')
plt.ylim([0, 100])
name = '3SGP'
plt.scatter(data[0]['value1'][:].values, data[0]['value2'][:].values, s=2, marker='.')
# plt.scatter(data[1]['value1'][:].values, data[1]['value2'][:].values, s=2, marker='.')
# plt.scatter(data[2]['value1'][:].values, data[2]['value2'][:].values, s=2, marker='.')
# plt.scatter(data[3]['value1'][:].values, data[3]['value2'][:].values, s=2, marker='.')
plt.xlabel('Ib/I0')
plt.ylabel('Dwell time(ms)')
plt.title(name)
plt.savefig(name+'.svg')
plt.savefig(name+'.png')

