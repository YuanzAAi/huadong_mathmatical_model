from matplotlib import pyplot as plt
from find_optimal_window_size import find_optimal_window_size
from find_window_period import find_window_period
from plot_regression_result import plot_regression_result
from data import data, columns
from sliding_window_regression import sliding_window_regression

# 选择多个变量进行滑动窗口回归分析，这里从问题1和问题2的线性或者非线性关系得到的模型进行滑动回归，x，y根据自己的需要选择性更换
#y为因变量，x为自变量
x = data[columns[1:4]]
y = data['Close_rb']


# 调用找出最优窗口长度函数，得到最优的窗口长度和AIC列表
optimal_window_size, aic_list = find_optimal_window_size(x, y)

# 打印最优的窗口长度
print(f'最优窗口长度为 {optimal_window_size} days.')

# 画出不同窗口长度下的AIC值的变化曲线
plt.figure(figsize=(8, 6))
plt.plot(range(10, 101, 10), aic_list)
plt.xlabel('Window Size')
plt.ylabel('AIC')
plt.title('AIC vs Window Size')
plt.grid()
plt.show()

# 设置滑动窗口的步长，例如1天
step_size = 1

# 调用滑动窗口回归函数，得到索引列表和结果列表
index_list, result_list = sliding_window_regression(x, y, optimal_window_size, step_size)

# 设置判据的阈值和权重，例如0.97
threshold = 0.97

# 调用找出窗口期函数，得到窗口期列表
window_period_list = find_window_period(result_list, threshold)

# 打印窗口期列表，并转换为日期格式
print('窗口期列表为:')
for start_index, end_index in window_period_list:
    start_date = data['Date_hc'].iloc[start_index]
    end_date = data['Date_hc'].iloc[end_index]
    print(f'{start_date} - {end_date}')

# 调用画出回归结果函数，得到变化曲线
plot_regression_result(index_list, result_list, columns)