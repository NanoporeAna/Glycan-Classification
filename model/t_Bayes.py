import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import datasets, metrics
file_name = ['../data/3SGP---63554 eventsBy3on2000.csv', '../data/3SLP---62910 eventsBy3on2000.csv',
             '../data/LSTaP---64097 eventsBy3on2000.csv', '../data/STetraP---63445 eventsBy3on2000.csv']
def load_svm_data(file_name):
    features = []
    labels = []
    for index, file in enumerate(file_name):
        raw_data = pd.read_csv(file, header=0)  # 读取csv数据，并将第一行视为表头，返回DataFrame类型
        length = raw_data.shape[0]
        data = raw_data.values
        features.extend(data[::, 1::])
        labels.extend([index]*length)
    return features, labels
if __name__ == '__main__':
    clf = joblib.load('naive_bayes_model.m')
    labels = ['3SGP', '3SLP', 'LSTaP', 'STetraP']
    fea, lab = load_svm_data(file_name)
    pre = clf.predict(fea)
    # 输出混淆矩阵
    confusion_matrix = metrics.confusion_matrix(lab, pre)
    # 画出混淆矩阵
    # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=labels)
    disp.plot()
    plt.title('naive_bayes')
    plt.savefig('naive_bayes.png')
    plt.show()

