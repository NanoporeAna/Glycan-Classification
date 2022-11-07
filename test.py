import os

import pandas as pd
from pandas.tests.io.excel.test_xlwt import xlwt

save_path = 't.xlsx'
if not os.path.exists(save_path):
    df = pd.DataFrame()  # 表示创建空的表
    df.to_excel(save_path)