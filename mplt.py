import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data = pd.read_json('HSQC_shuffle_tags_rdkit.json')
t = data['ExactHSQC'].values
tag = data['tags'].values
# def get_hsqc_tags_tensor(hsqc_nmr, tags, c_scale=5, h_scale=20, c_min_value=-50, c_max_value=350, h_min_value=-4,
#                          h_max_value=16):
#     """
#
#     :param self:
#     :param hsqc_nmr:
#     :param tags C 多重性标签
#     :param c_scale: 0.2ppm
#     :param h_scale: 0.05ppm
#     :param c_min_value:
#     :param c_max_value:
#     :param h_min_value:
#     :param h_max_value:
#     :return: 灰度二值图像
#     """
#     cunits = int((c_max_value - c_min_value) * c_scale)
#     hunits = (h_max_value - h_min_value) * h_scale
#
#     data = [(round(((value[1]) - c_min_value) * c_scale), round(((value[0]) - h_min_value) * h_scale)) for
#             value in hsqc_nmr]
#     # 初始化单张图片的格式[800,400]shape的0矩阵
#     C_tags1, C_tags2, C_tags3 = [], [], []
#     for inx, ch in enumerate(data):
#         a, b = ch[0], hunits - ch[1]
#         if tags[inx]==1:
#             C_tags1.append((a,b))
#         if tags[inx]==2:
#             C_tags2.append((a,b))
#         if tags[inx]==3:
#             C_tags3.append((a,b))
#     return data, C_tags1, C_tags2, C_tags3
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


C1, C2, C3,H1,H2,H3 = [], [], [], [], [], []
for idx, val in enumerate(t):
    C_tags1, C_tags2, C_tags3, h_tags1, h_tags2, h_tags3 = get_hsqc_tags_tensor(val, tag[idx])
    if C_tags1:
        C1.extend(C_tags1)
        H1.extend(h_tags1)
    if C_tags2:
        C2.extend(C_tags2)
        H2.extend(h_tags2)
    if C_tags3:
        C3.extend(C_tags3)
        H3.extend(h_tags3)
# dataf = {'C1':C1, 'H1':H1}
# data2 = {'C2':C2, 'H2':H2}
# data3 = {'C3':C3, 'H3':H3}
# df1 = pd.DataFrame(dataf)
# df2 = pd.DataFrame(data2)
# df3 = pd.DataFrame(data3)
# 文件内容太多，无法保存
# with pd.csWriter('CH.xlsx') as writer:
#     df1.to_excel(writer, sheet_name='CH1', index=False)
#     df2.to_excel(writer, sheet_name='CH1', index=False)
#     df3.to_excel(writer, sheet_name='CH1', index=False)
# 改用保存至csv
# df1.to_csv('Ch1.csv', index=False)
# df2.to_csv('Ch2.csv', index=False)
# df3.to_csv('Ch3.csv', index=False)

plt.style.use('classic')
fig, ax = plt.subplots(figsize=(20, 16), dpi=300)
dian = plt.scatter(H1, C1, c='k', s=5)
fontdict1={'size':17, 'color': 'k', 'family': 'Times New Roman'}
ax.set_xlabel('H', fontdict=fontdict1)
ax.set_xlim((-4, 13))
ax.invert_xaxis()
ax.set_ylabel('C', fontdict=fontdict1)
ax.yaxis.set_ticks_position('right')
ax.set_ylim((-20, 220))
ax.invert_yaxis()
nbin = 400
H, xedges, yedges = np.histogram2d(H1, C1, bins=nbin)
H = np.rot90(H)
H = np.flipud(H)
Hmasked = np.ma.masked_where(H==0,H)
plt.pcolormesh(xedges, yedges, Hmasked, cmap='jet', vmin=0, vmax=160)
cbar = plt.colorbar(ax=ax, ticks=[0,20,40,60,80,100,120,140,160], drawedges=False)
#cbar.ax.set_ylabel('Frequency',fontdict=colorbarfontdict)
cbar.ax.set_title('Counts', fontdict=fontdict1,pad=8)
cbar.ax.tick_params(labelsize=12, direction='in')
cbar.ax.set_yticklabels(['0','20','40','60','80','100','120','140','>160'], family='Times New Roman')
plt.title('CH1', fontdict=fontdict1)
plt.savefig('scatter_ch1.png', format='png', dpi=800, bbox_inches='tight', transparent=True)
plt.show()



# C = []
# H = []
# for i in Ch_tags1:
#     C.append(i[0])
#     H.append(i[1])
# plt.scatter(x=H, y=C, s=4, c='r', label='Ch1')
# C = []
# H = []
# for i in Ch_tags2:
#     C.append(i[0])
#     H.append(i[1])
# plt.scatter(x=H, y=C, s=4, c='g', label='Ch2')
# C = []
# H = []
# for i in Ch_tags3:
#     C.append(i[0])
#     H.append(i[1])
# plt.scatter(x=H, y=C, s=4, c='b', label='Ch3')
# plt.legend()
# plt.savefig('tt.jpg')
# plt.savefig('tt.svg')
# plt.show()

