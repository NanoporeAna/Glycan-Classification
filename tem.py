import matplotlib.pyplot as plt
import pyabf
import torch
import sklearn
# abf = pyabf.ABF('./data/6SLP-10min.abf')
# # abf.setSweep(sweepNumber=3, channel=0)
# print(abf.sweepY.shape)
# print(abf.sweepX.shape)
# print(abf.sweepC.shape)
# plt.plot(abf.sweepX, abf.sweepY)
# plt.show()
x= torch.randn([3, 4])
mask = (x == x.max(dim=1, keepdim=True)[0]).to(dtype=torch.int32)
print(mask)