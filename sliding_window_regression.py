# 定义一个函数，用来计算多个变量之间的滑动窗口回归结果
from linear_regression import linear_regression

def sliding_window_regression(x, y, window_size, step_size):
    # 初始化一个空列表，用来存储滑动窗口的起始和结束索引
    index_list = []
    # 初始化一个空列表，用来存储滑动窗口的回归结果
    result_list = []
    # 获取数据的长度
    n = len(x)
    # 计算滑动窗口的数量
    num_windows = (n - window_size) // step_size + 1
    # 循环遍历每个滑动窗口
    for i in range(num_windows):
        # 计算滑动窗口的起始和结束索引
        start = i * step_size
        end = start + window_size
        # 将索引添加到索引列表中
        index_list.append((start, end))
        # 截取滑动窗口内的数据
        x_window = x[start:end]
        y_window = y[start:end]
        # 调用线性回归函数或者非线性回归函数，计算滑动窗口内的回归结果
        params, rsquared, aic= linear_regression(x_window, y_window) # 或者 nonlinear_regression(x_window, y_window)
        # 将回归结果添加到结果列表中
        result_list.append((params, rsquared, aic))
        # 返回索引列表和结果列表
    return index_list, result_list