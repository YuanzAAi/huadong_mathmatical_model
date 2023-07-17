import numpy as np
# 定义一个函数，将超出上下界的值替换为中位数
def replace_outliers_with_median(data, column):
    median = data[column].median()
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    data[column] = np.where((data[column] > upper_bound) | (data[column] < lower_bound), median, data[column])
    return data