import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from data import data
from problem4_function import log_return, minimize_risk, maximize_sharpe_ratio, portfolio_performance, sharpe_ratio, \
    kelly_criterion

# 定义一些参数和约束条件
columns = data.columns[1:10:2]
n = len(columns) # 品种数量,columns未必要全部选择，可以选择性地挑选一些品种进行投资
risk_free_rate = 0.01 # 无风险收益率，可以根据实际情况修改
window = 10 # 滑动窗口长度，可以根据实际情况修改
bounds = tuple((0.0, 1.0) for i in range(n)) # 投资比例的范围
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # 投资比例的和为1

# 计算每个品种的对数收益率，并存储在一个新的数据框中
log_returns = pd.DataFrame()
for column in columns:
    log_returns[column] = log_return(data[column])
log_returns.head()

# 初始化一个空的数据框，用于存储每个滑动窗口下的最优投资比例、收益率、风险、夏普比率和凯利公式
optimal_portfolio = pd.DataFrame(columns=['Date', 'Weights', 'Return', 'Risk', 'Sharpe', 'Kelly'])

# 对每个滑动窗口进行循环
for i in range(window, len(data)):
    # 获取当前窗口下的数据子集
    subset = log_returns.iloc[i-window:i]
    # 计算当前窗口下的每个品种的平均收益率和协方差矩阵
    mean_returns = subset.mean()
    cov_matrix = subset.cov()
    # 使用最小二乘法求解最优投资比例，使得风险最小
    min_result = minimize(minimize_risk, n*[1./n], args=(mean_returns, cov_matrix), method='SLSQP', bounds=bounds, constraints=constraints)
    # 使用最小二乘法求解最优投资比例，使得夏普比率最大
    max_result = minimize(maximize_sharpe_ratio, n*[1./n], args=(mean_returns, cov_matrix, risk_free_rate), method='SLSQP', bounds=bounds, constraints=constraints)
    # 获取最优投资比例、收益率、风险、夏普比率和凯利公式
    max_weights = max_result.x
    max_return, max_risk = portfolio_performance(max_weights, mean_returns, cov_matrix)
    max_sharpe = sharpe_ratio(max_return, risk_free_rate, max_risk)
    max_kelly = kelly_criterion(max_return, risk_free_rate, max_risk)
    # 获取当前窗口的最后一个日期
    date = data.iloc[i]['Date_hc']
    # 将当前窗口下的最优投资比例、收益率、风险、夏普比率和凯利公式存储到数据框中
    optimal_portfolio = optimal_portfolio.append({'Date': date,
                                                  'Weights': max_weights,
                                                  'Return': max_return,
                                                  'Risk': max_risk,
                                                  'Sharpe': max_sharpe,
                                                  'Kelly': max_kelly}, ignore_index=True)

# 设置日期为索引
optimal_portfolio.set_index('Date', inplace=True)


# 绘制最优投资组合的收益率和风险的散点图，并用颜色表示夏普比率
plt.figure(figsize=(10, 6))
plt.scatter(optimal_portfolio['Risk'], optimal_portfolio['Return'], c=optimal_portfolio['Sharpe'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Risk')
plt.ylabel('Return')
plt.title('Optimal Portfolio Performance')
plt.show()

# 绘制最优投资组合的凯利公式的折线图
plt.figure(figsize=(10, 6))
plt.plot(optimal_portfolio['Kelly'])
plt.xlabel('Date')
plt.ylabel('Kelly Criterion')
plt.title('Optimal Portfolio Kelly Criterion')
plt.show()

# 绘制每个品种在最优投资组合中的权重变化的堆叠面积图
plt.figure(figsize=(10, 6))
weights = optimal_portfolio['Weights'].apply(np.array)
weights = pd.DataFrame(weights.tolist(), index=weights.index)
plt.stackplot(optimal_portfolio.index,weights.T, labels=columns)
plt.legend(loc='upper left')
plt.xlabel('Date')
plt.ylabel('Weights')
plt.title('Optimal Portfolio Weights')
plt.show()

# 假设初始资金为10000元，根据每个品种在最优投资组合中的权重，计算出每个品种应该买入或卖出多少金额
# 例如，如果某个品种在某一天的权重是0.2，那么应该用10000*0.2=2000元来买入这个品种
# 如果第二天这个品种的权重变成了0.3，那么应该用10000*0.1=1000元来追加买入这个品种
# 如果第三天这个品种的权重变成了0.1，那么应该用10000*0.2=2000元来卖出这个品种
signals = weights * 10000 # 计算每个品种每天应该持有多少金额
signals = signals.diff() # 计算每个品种每天应该买入或卖出多少金额
signals = signals.fillna(0) # 填充第一天的空值为0

# 绘制每个品种每天应该买入或卖出多少金额的折线图，使用matplotlib的plot函数，x轴是日期，y轴是金额，labels是品种的名称
plt.plot(signals.index, signals.to_numpy())

# 添加图例、标题、坐标轴标签等
plt.legend(columns,loc='upper left')
plt.title('buy or sell signals')
plt.xlabel('date')
plt.ylabel('money')
plt.grid()
plt.show()

#输出到Excel上
with pd.ExcelWriter('最优投资比率和每天投资多少钱.xlsx') as writer:
    optimal_portfolio.to_excel(writer, sheet_name='Sheet1')
    signals.to_excel(writer, sheet_name='Sheet2')