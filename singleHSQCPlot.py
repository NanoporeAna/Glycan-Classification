import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_json('HSQC_shuffle_tags_rdkit.json')
t = data['ExactHSQC'].values
tag = data['tags'].values
H, C = [], []
for i in t[666]:
    H.append(i[0])
    C.append(i[1])
def get_hsqc_tags_tensor(hsqc_nmr, tags):
    C1, C2, C3,H1,H2,H3 = [], [], [], [], [], []
    for idx, val in enumerate(hsqc_nmr):
        if tags[idx]==1:
            C1.append(val[1])
            H1.append(val[0])
        if tags[idx]==2:
            C2.append(val[1])
            H2.append(val[0])
        if tags[idx]==3:
            C3.append(val[1])
            H3.append(val[0])
    return C1, C2, C3,H1,H2,H3

C_tags1, C_tags2, C_tags3, h_tags1, h_tags2, h_tags3 = get_hsqc_tags_tensor(t[666], tag[666])

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
dian = plt.scatter(h_tags3, C_tags3, c='g', s=5)
fontdict1={'size':17, 'color': 'k', 'family': 'Times New Roman'}
ax.set_xlabel('H', fontdict=fontdict1)
ax.set_xlim((-4, 16))
ax.invert_xaxis()
ax.set_ylabel('C', fontdict=fontdict1)
ax.yaxis.set_ticks_position('right')
ax.set_ylim((-50, 350))
ax.invert_yaxis()
plt.title('CH3')
plt.savefig('sample3.png',format='png', dpi=600, bbox_inches='tight', transparent=True)