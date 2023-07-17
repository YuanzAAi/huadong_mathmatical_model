from matplotlib import pyplot as plt
from data import data

# 定义一个函数，用来画出回归结果的变化曲线
def plot_regression_result(index_list, result_list, columns):
    fig = plt.figure(figsize=(12, 8))
    # 设置标题
    fig.suptitle('Sliding Window Regression Result')
    # 循环遍历每个回归结果
    for i, (params, rsquared, aic) in enumerate(result_list):
        # 获取滑动窗口的起始和结束索引
        start, end = index_list[i]
        # 获取滑动窗口的中间索引
        mid = (start + end) // 2

        # 画出回归系数和决定系数点，以滑动窗口的中间索引为横坐标，以回归系数和决定系数作为纵坐标

        # 循环遍历每个回归参数，并用不同颜色表示不同品种对应的回归系数
        plt.figure(1)
        plt.grid()
        plt.ylim(-10, 10)
        plt.xticks(ticks=data.index[::365], labels=data['Date_hc'].dt.year[::365])
        for j in range(len(params)):
            plt.scatter(mid, params[j], color=plt.get_cmap('tab10')(j / len(params)),
                        label=f'Regression Coefficient of {columns[j]}' if i == 0 else '')
        plt.legend()
        # 用红色表示决定系数
        plt.figure(2)
        plt.grid()
        plt.ylim(-1, 1)
        plt.xticks(ticks=data.index[::365], labels=data['Date_hc'].dt.year[::365])
        plt.scatter(mid,rsquared, color='red', label='R-squared' if i == 0 else '')
        plt.legend()
    plt.show()
