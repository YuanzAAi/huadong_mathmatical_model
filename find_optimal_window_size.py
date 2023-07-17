# 定义一个函数，用来根据AIC来确定最优的窗口长度
import numpy as np
from sliding_window_regression import sliding_window_regression

def find_optimal_window_size(x, y):
    # 初始化一个空列表，用来存储不同窗口长度下的AIC值
    aic_list = []
    # 初始化一个变量，用来记录最小的AIC值
    min_aic = np.inf
    # 初始化一个变量，用来记录最优的窗口长度
    optimal_window_size = None
    # 循环遍历不同的窗口长度，例如从10天到100天，每隔10天取一个值
    for window_size in range(10, 101, 10):
        # 调用滑动窗口回归函数，得到索引列表和结果列表
        index_list, result_list = sliding_window_regression(x, y, window_size, 1)
        # 计算所有滑动窗口的AIC值的平均值，并添加到AIC列表中
        aic_mean = np.mean([result[2] for result in result_list])
        aic_list.append(aic_mean)
        # 如果当前的AIC值小于最小的AIC值，那么更新最小的AIC值和最优的窗口长度
        if aic_mean < min_aic:
            min_aic = aic_mean
            optimal_window_size = window_size

    # 返回最优的窗口长度和AIC列表
    return optimal_window_size, aic_list