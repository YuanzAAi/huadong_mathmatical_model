# 定义一个函数，用来计算多个变量之间的非线性回归结果
import numpy as np
from scipy.optimize import curve_fit


def nonlinear_regression(x, y):
    # 将 x 和 y 转换为 numpy 数组
    x = np.array(x)
    y = np.array(y)
    # 定义一个非线性函数，例如二次函数
    def func(x, a,  c):
        return a *  x**2 + c
    # 使用最小二乘法拟合非线性模型
    popt, pcov = curve_fit(func,x, y)
    # 计算预测值和残差
    y_pred = func(x, *popt)
    resid = y - y_pred
    # 计算回归系数，决定系数，AIC和均方误差
    params = np.append(popt, np.nan)
    ss_res = np.sum(resid**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    rsquared = 1 - (ss_res / ss_tot)
    n = len(y)
    k = len(popt) + 1
    aic = n * np.log(ss_res / n) + 2 * k
    mse_model = ss_res / (n - k)
    # 返回回归系数，决定系数，AIC和均方误差
    return params, rsquared, aic, mse_model