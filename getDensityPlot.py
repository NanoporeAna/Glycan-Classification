import matplotlib.pyplot as plt
import numpy as np

class DensityPlot():
    def __init__(self, datax, datay, bins, xlabel, ylabel, title, filename):
        self.datax = datax
        self.datay = datay
        self.bins = bins
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.title = title
        self.filename = filename
    def getHSQCDensityPlot(self):
        plt.style.use('classic')
        fig, ax = plt.subplots(figsize=(20, 16), dpi=300)
        dian = plt.scatter(self.datax, self.datay, c='k', s=5)
        fontdict1 = {'size': 17, 'color': 'k', 'family': 'Times New Roman'}
        ax.set_xlabel(self.datax, fontdict=fontdict1)
        ax.set_xlim((-4, 12))
        ax.set_ylabel(self.ylabel, fontdict=fontdict1)
        ax.set_ylim((-20, 220))
        H, xedges, yedges = np.histogram2d(self.datax, self.datay, bins=self.bins)
        H = np.rot90(H)
        H = np.flipud(H)
        h_masked = np.ma.masked_where(H == 0, H)
        plt.pcolormesh(xedges, yedges, h_masked, cmap='jet', vmin=0, vmax=160)
        cbar = plt.colorbar(ax=ax, ticks=[0, 20, 40, 60, 80, 100, 120, 140, 160], drawedges=False)
        cbar.ax.set_title('Counts', fontdict=fontdict1, pad=8)
        cbar.ax.tick_params(labelsize=12, direction='in')
        cbar.ax.set_yticklabels(['0', '20', '40', '60', '80', '100', '120', '140', '>160'], family='Times New Roman')
        plt.title(self.title, fontdict=fontdict1)
        plt.savefig(self.filename, dpi=800, bbox_inches='tight')

import pandas as pd
data = pd.read_json('HSQC_shuffle_tags_rdkit.json')
t = data['ExactHSQC'].values
tag = data['tags'].values
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
t = DensityPlot(H1,C1, 400, 'H','C','CH1','test.png')
t.getHSQCDensityPlot()