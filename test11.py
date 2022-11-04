import numpy as np
from sklearn.metrics import recall_score

labels = [1, 2, 3, 4] # 有哪几类

pred = [1, 2, 3, 4, 2, 3, 4, 1] # 预测的值

target = [2, 3, 1, 4, 1, 4, 4, 1] # 真实的值


r = recall_score(pred, target, labels=labels, average="macro")

print(r)

print(np.mean(np.array(pred) == np.array(target)))
