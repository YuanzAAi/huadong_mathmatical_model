from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn import metrics
from itertools import combinations
from data import data, columns

# 创建一个空的列表，用来存储每个模型的均方误差和决定系数
mse_r2_list = []

# 对每个品种，不断增加其他品种作为自变量，建立多元LSTM模型，并计算均方误差和决定系数
for i in range(len(columns)):
    y = data[columns[i]].values.reshape(-1, 1) # 因变量
    x_columns = columns.drop(columns[i]) # 剔除因变量对应的列名
    for j in range(1, len(x_columns) + 1):
        for x_comb in combinations(x_columns, j): # 遍历所有可能的自变量组合
            x = data[list(x_comb)].values # 自变量
            scaler = MinMaxScaler(feature_range=(0, 1))  # 创建一个MinMaxScaler对象，将数据缩放到 [0, 1] 区间内
            x = scaler.fit_transform(x)  # 对自变量进行归一化
            y = scaler.fit_transform(y)  # 对因变量进行归一化
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) # 数据划分
            x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1])) # 调整训练集数据的形状，以适应LSTM模型的输入要求
            x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1])) # 调整测试集数据的形状，以适应LSTM模型的输入要求
            model = keras.Sequential() # 创建一个序贯模型对象
            model.add(keras.layers.LSTM(50, input_shape=(x_train.shape[1], x_train.shape[2]))) # 添加一个LSTM层，设置隐藏单元为50，输入形状为(1, 自变量个数)
            model.add(keras.layers.Dense(1)) # 添加一个全连接层，设置输出单元为1，即因变量的预测值
            model.compile(loss='mse', optimizer='adam') # 编译模型，设置损失函数为均方误差，优化器为Adam算法
            model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0) # 拟合训练集数据，设置迭代次数为50，批次大小为32，不打印训练过程信息
            y_pred = model.predict(x_test) # 预测测试集数据
            y_test = scaler.inverse_transform(y_test)  # 对y_test进行反归一化
            y_pred = scaler.inverse_transform(y_pred)  # 对y_pred进行反归一化
            mse = metrics.mean_squared_error(y_test, y_pred) # 计算均方误差
            r2 = metrics.r2_score(y_test, y_pred) # 计算决定系数
            mse_r2_list.append((mse, r2, columns[i], list(x_comb))) # 将均方误差、决定系数和自变量存入列表

# 对列表按照均方误差从小到大进行排序
mse_r2_list.sort(key=lambda x: x[0])
model_nonlinear_LSTM = mse_r2_list[0:5]

# 打印均方误差最小的前五个模型以及其参数和决定系数
for i in range(5):
    print('第{}个模型：'.format(i + 1))
    print('因变量：', mse_r2_list[i][2])
    print('自变量：', mse_r2_list[i][3])
    print('均方误差：', mse_r2_list[i][0])
    print('决定系数：', mse_r2_list[i][1])
    print('---------------------分割线')

# 将模型保存到Excel文件
model_nonlinear_LSTM.to_excel('由多元LSTM模型得到的非线性关系.xlsx')

