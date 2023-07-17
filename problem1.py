import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from itertools import combinations
from replace_outliers_with_median import replace_outliers_with_median

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

# 查看数据基本信息
print(data.info())

# 设置显示所有行和列
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# 查看数据描述性统计
data_describe = (data.describe(datetime_is_numeric=True))

# 绘制数据的箱线图，观察异常值
data.boxplot()
plt.xticks(rotation=90)
plt.show()

# 绘制数据的折线图，观察变化趋势
fig, axs = plt.subplots(3, 3, figsize=(20, 15),dpi = 95)

axs[0, 0].plot(data['Date_hc'], data['Close_hc'])
axs[0, 0].set_title('hc',loc='center',fontsize=16, color='blue')
axs[0, 1].plot(data['Date_i'], data['Close_i'])
axs[0, 1].set_title('i',loc='center',fontsize=16, color='blue')
axs[0, 2].plot(data['Date_j'], data['Close_j'])
axs[0, 2].set_title('j',loc='center',fontsize=16, color='blue')
axs[1, 0].plot(data['Date_jm'], data['Close_jm'])
axs[1, 0].set_title('jm',loc='center',fontsize=16, color='blue')
axs[1, 1].plot(data['Date_rb'], data['Close_rb'])
axs[1, 1].set_title('rb',loc='center',fontsize=16, color='blue')
axs[1, 2].plot(data['Date_sf'], data['Close_sf'])
axs[1, 2].set_title('sf',loc='center',fontsize=16, color='blue')
axs[2, 0].plot(data['Date_sm'], data['Close_sm'])
axs[2, 0].set_title('sm',loc='center',fontsize=16, color='blue')
axs[2, 1].plot(data['Date_ss'], data['Close_ss'])
axs[2, 1].set_title('ss',loc='center',fontsize=16, color='blue')
axs[2, 2].plot(data['Date_wr'], data['Close_wr'])
axs[2, 2].set_title('wr',loc='center',fontsize=16, color='blue')

for ax in axs.flat:
    ax.set(xlabel='', ylabel='')

plt.tight_layout()
plt.show()


# 针对每个列进行异常值替换
data = replace_outliers_with_median(data, 'Close_sf')
data = replace_outliers_with_median(data, 'Close_sm')
data = replace_outliers_with_median(data, 'Close_ss')

# 异常值替换后，再次绘制数据的箱线图，观察
data.boxplot()
plt.xticks(rotation=90)
plt.show()

# 异常值替换后，绘制数据的折线图，观察变化趋势
fig, axs = plt.subplots(3, 3, figsize=(20, 15),dpi = 95)

axs[0, 0].plot(data['Date_hc'], data['Close_hc'])
axs[0, 0].set_title('hc',loc='center',fontsize=16, color='blue')
axs[0, 1].plot(data['Date_i'], data['Close_i'])
axs[0, 1].set_title('i',loc='center',fontsize=16, color='blue')
axs[0, 2].plot(data['Date_j'], data['Close_j'])
axs[0, 2].set_title('j',loc='center',fontsize=16, color='blue')
axs[1, 0].plot(data['Date_jm'], data['Close_jm'])
axs[1, 0].set_title('jm',loc='center',fontsize=16, color='blue')
axs[1, 1].plot(data['Date_rb'], data['Close_rb'])
axs[1, 1].set_title('rb',loc='center',fontsize=16, color='blue')
axs[1, 2].plot(data['Date_sf'], data['Close_sf'])
axs[1, 2].set_title('sf',loc='center',fontsize=16, color='blue')
axs[2, 0].plot(data['Date_sm'], data['Close_sm'])
axs[2, 0].set_title('sm',loc='center',fontsize=16, color='blue')
axs[2, 1].plot(data['Date_ss'], data['Close_ss'])
axs[2, 1].set_title('ss',loc='center',fontsize=16, color='blue')
axs[2, 2].plot(data['Date_wr'], data['Close_wr'])
axs[2, 2].set_title('wr',loc='center',fontsize=16, color='blue')

for ax in axs.flat:
    ax.set(xlabel='', ylabel='')

plt.tight_layout()
plt.show()

# 计算收盘价之间的相关系数矩阵
cor = data.corr(method='pearson')

# 绘制热力图，观察线性关系
plt.figure(figsize=(10, 10))
sns.heatmap(cor, annot=True, center=0.5, fmt='.2f', linewidth=0.5, vmin=-1, vmax=1, xticklabels=True, yticklabels=True, square=True, cbar=True, cmap='coolwarm')
plt.show()

# 删除所有含有缺失值的行
data.dropna(inplace=True)


# 获取收盘价的列名
columns = data.columns[1:18:2]

# 创建空的矩阵，用来存储回归系数、p值和决定系数
coef_matrix = np.zeros((len(columns), len(columns)))
p_matrix = np.zeros((len(columns), len(columns)))
r2_matrix = np.zeros((len(columns), len(columns)))

# 对每个品种，使用其他品种的价格数据作为自变量，建立线性回归模型，并得出相应的回归系数、p值和决定系数
for i in range(len(columns)):
    y = data[columns[i]].values.reshape(-1, 1) # 因变量
    for j in range(i + 1, len(columns)):
        x = data[columns[j]].values.reshape(-1, 1) # 自变量
        slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y.flatten()) # 使用最小二乘法进行线性回归
        coef_matrix[i][j] = slope # 存储回归系数
        p_matrix[i][j] = p_value # 存储p值
        r2_matrix[i][j] = r_value ** 2 # 存储决定系数

# 绘制热力图，显示矩阵结果
plt.figure(figsize=(10, 10))
sns.heatmap(coef_matrix, annot=True, center=0.5, fmt='.2f', linewidth=0.5, vmin=-1, vmax=1, xticklabels=columns, yticklabels=columns, square=True, cbar=True, cmap='coolwarm')
plt.title('coef_SimpleLinearRegression')
plt.show()

plt.figure(figsize=(10, 10))
sns.heatmap(p_matrix, annot=True, center=0.5, fmt='.2e', linewidth=0.5, vmin=0, vmax=1, xticklabels=columns, yticklabels=columns, square=True, cbar=True, cmap='coolwarm')
plt.title('p-Value')
plt.show()

plt.figure(figsize=(10, 10))
sns.heatmap(r2_matrix, annot=True, center=0.5, fmt='.2f', linewidth=0.5, vmin=0, vmax=1, xticklabels=columns, yticklabels=columns, square=True, cbar=True, cmap='coolwarm')
plt.title('R-square')
plt.show()

# 创建一个空的列表，用来存储每个模型的均方误差和参数
mse_r2_list = []

# 对每个品种，不断增加其他品种作为自变量，建立多元线性回归模型，并计算均方误差
for i in range(len(columns)):
    y = data[columns[i]].values.reshape(-1, 1) # 因变量
    x_columns = columns.drop(columns[i]) # 剔除因变量对应的列名
    for j in range(1, len(x_columns) + 1):
        for x_comb in combinations(x_columns, j): # 遍历所有可能的自变量组合
            x = data[list(x_comb)].values # 自变量
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) # 数据划分
            model = LinearRegression() # 创建模型对象
            model.fit(x_train, y_train) # 拟合训练集数据
            y_pred = model.predict(x_test) # 预测测试集数据
            mse = metrics.mean_squared_error(y_test, y_pred) # 计算均方误差
            r2 = metrics.r2_score(y_test, y_pred)  # 计算决定系数
            mse_r2_list.append((mse, r2, columns[i], list(x_comb), model.intercept_, model.coef_))

# 对列表按照均方误差从小到大进行排序
mse_r2_list.sort(key=lambda x: x[0])
modelliner = mse_r2_list[0:10]

# 输出均方误差最小的前十个模型以及其参数
for i in range(10):
    print('第{}个模型：'.format(i + 1))
    print('因变量：', mse_r2_list[i][2])
    print('自变量：', mse_r2_list[i][3])
    print('截距：', mse_r2_list[i][4])
    print('回归系数：', mse_r2_list[i][5])
    print('均方误差：', mse_r2_list[i][0])
    print('决定系数：', mse_r2_list[i][1])
    print('---------------------分割线')

with pd.ExcelWriter('描述性统计和线性关系模型.xlsx') as writer:
    data_describe.to_excel(writer, sheet_name='Sheet1')