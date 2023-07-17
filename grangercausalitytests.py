import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

from data import columns, data

granger_results = []
for i in range(len(columns)):
    for j in range(i + 1, len(columns)):
        x = data[[columns[i], columns[j]]].values  # 选择两个品种的收盘价作为数据
        result = grangercausalitytests(x, maxlag=5)  # 进行Granger因果分析，设置最大滞后阶数为5
        granger_results.append({
            '品种1': columns[i],
            '品种2': columns[j],
            '滞后阶数1_F统计量': result[1][0]['ssr_ftest'][0],
            '滞后阶数1_p值': result[1][0]['ssr_ftest'][1],
            '滞后阶数2_F统计量': result[2][0]['ssr_ftest'][0],
            '滞后阶数2_p值': result[2][0]['ssr_ftest'][1],
            '滞后阶数3_F统计量': result[3][0]['ssr_ftest'][0],
            '滞后阶数3_p值': result[3][0]['ssr_ftest'][1],
            '滞后阶数4_F统计量': result[4][0]['ssr_ftest'][0],
            '滞后阶数4_p值': result[4][0]['ssr_ftest'][1],
            '滞后阶数5_F统计量': result[5][0]['ssr_ftest'][0],
            '滞后阶数5_p值': result[5][0]['ssr_ftest'][1],
        })

# 将granger_results转换为DataFrame对象
granger_df = pd.DataFrame(granger_results)

# 保存到Excel
granger_df.to_excel('格兰杰因果分析.xlsx', index=False)