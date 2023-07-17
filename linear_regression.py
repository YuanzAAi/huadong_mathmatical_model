# 定义一个函数，用来计算多个变量之间的线性回归结果
import numpy as np
import statsmodels.api as sm

def linear_regression(x, y):
    # 将 X 和 y 转换为 numpy 数组
    x = np.array(x)
    y = np.array(y)
    # 添加一个常数项
    x = sm.add_constant(x)
    # 使用最小二乘法拟合线性模型
    model = sm.OLS(y, x)
    results = model.fit()
    # 返回回归系数，决定系数，AIC
    return results.params, results.rsquared, results.aic