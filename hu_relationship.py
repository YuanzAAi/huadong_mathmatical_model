import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from replace_outliers_with_median import replace_outliers_with_median
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据文件
hc2301 = pd.read_excel('hc2301.xls', parse_dates=['日期'])
i2301 = pd.read_excel('i2301.xls', parse_dates=['日期'])
j2301 = pd.read_excel('j2301.xls', parse_dates=['日期'])
jm2301 = pd.read_excel('jm2301.xls', parse_dates=['日期'])
rb2301 = pd.read_excel('rb2301.xls', parse_dates=['日期'])
sf301 = pd.read_excel('sf301.xls', parse_dates=['日期'], dtype={'收盘价': 'float64'}, thousands=',')
sm301 = pd.read_excel('sm301.xls', parse_dates=['日期'], dtype={'收盘价': 'float64'}, thousands=',')
ss2301 = pd.read_excel('ss2301.xls', parse_dates=['日期'])
wr2301 = pd.read_excel('wr2301.xls', parse_dates=['日期'])

# 使用 pd.concat 函数合并数据框
data = pd.concat([hc2301[['日期','收盘价']].rename(columns={'日期':'Date_hc','收盘价': 'Close_hc'}),
                  i2301[['日期','收盘价']].rename(columns={'日期':'Date_i','收盘价': 'Close_i'}),
                  j2301[['日期','收盘价']].rename(columns={'日期':'Date_j','收盘价': 'Close_j'}),
                  jm2301[['日期','收盘价']].rename(columns={'日期':'Date_jm','收盘价': 'Close_jm'}),
                  rb2301[['日期','收盘价']].rename(columns={'日期':'Date_rb','收盘价': 'Close_rb'}),
                  sf301[['日期','收盘价']].rename(columns={'日期':'Date_sf','收盘价': 'Close_sf'}),
                  sm301[['日期','收盘价']].rename(columns={'日期':'Date_sm','收盘价': 'Close_sm'}),
                  ss2301[['日期','收盘价']].rename(columns={'日期':'Date_ss','收盘价': 'Close_ss'}),
                  wr2301[['日期','收盘价']].rename(columns={'日期':'Date_wr','收盘价': 'Close_wr'})],
                 axis=1)

# 针对每个列进行异常值替换
data = replace_outliers_with_median(data, 'Close_sf')
data = replace_outliers_with_median(data, 'Close_sm')
data = replace_outliers_with_median(data, 'Close_ss')

# 删除所有含有缺失值的行
data.dropna(inplace=True)

# 获取收盘价的列名
columns = data.columns[1:18:2]

# 等频分箱
k = 10 # 假设将每个品种的日线价格数据分成 10 个区间
discretizer = KBinsDiscretizer(n_bins=k, encode='ordinal', strategy='quantile') # 创建等频分箱器
close_discrete = discretizer.fit_transform(data[columns]) # 对价格数据进行等频分箱


# 计算互信息矩阵
n = 9 #有9种
C = np.zeros((n,n)) # 创建一个9*9的零矩阵
for i in range(n):
    for j in range(n):
        C[i,j] = mutual_info_score(close_discrete[:, i], close_discrete[:, j]) # 计算第 i 个品种和第 j 个品种之间的互信息


# 绘制热力图，观察互信息
plt.figure(figsize=(10, 10))
sns.heatmap(C, annot=True, center=0.5, fmt='.2f', linewidth=0.5, vmin=-1, vmax=1, xticklabels=columns, yticklabels=columns, square=True, cbar=True, cmap='coolwarm')
plt.title('hu-relationship')
plt.show()