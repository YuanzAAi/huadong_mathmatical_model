import pandas as pd
from replace_outliers_with_median import replace_outliers_with_median

# 读取数据文件
hc2301 = pd.read_excel('hc2301.xls', parse_dates=['日期'])
i2301 = pd.read_excel('i2301.xls', parse_dates=['日期'])
j2301 = pd.read_excel('j2301.xls', parse_dates=['日期'])
jm2301 = pd.read_excel('jm2301.xls', parse_dates=['日期'])
rb2301 = pd.read_excel('rb2301.xls', parse_dates=['日期'])
sf301 = pd.read_excel('sf301.xls', parse_dates=['日期'], dtype={'收盘价': 'float64'}, thousands=',')
sm301 = pd.read_excel('sm301.xls', parse_dates=['日期'], dtype={'收盘价': 'float64'}, thousands=',')
ss2301 = pd.read_excel('ss2301.xls', parse_dates=['日期'])
wr2301 = pd.read_excel('wr2301.xls', parse_dates=['日期'])

# 使用 pd.concat 函数合并数据框
data = pd.concat([hc2301[['日期','收盘价']].rename(columns={'日期':'Date_hc','收盘价': 'Close_hc'}),
                  i2301[['日期','收盘价']].rename(columns={'日期':'Date_i','收盘价': 'Close_i'}),
                  j2301[['日期','收盘价']].rename(columns={'日期':'Date_j','收盘价': 'Close_j'}),
                  jm2301[['日期','收盘价']].rename(columns={'日期':'Date_jm','收盘价': 'Close_jm'}),
                  rb2301[['日期','收盘价']].rename(columns={'日期':'Date_rb','收盘价': 'Close_rb'}),
                  sf301[['日期','收盘价']].rename(columns={'日期':'Date_sf','收盘价': 'Close_sf'}),
                  sm301[['日期','收盘价']].rename(columns={'日期':'Date_sm','收盘价': 'Close_sm'}),
                  ss2301[['日期','收盘价']].rename(columns={'日期':'Date_ss','收盘价': 'Close_ss'}),
                  wr2301[['日期','收盘价']].rename(columns={'日期':'Date_wr','收盘价': 'Close_wr'})],
                 axis=1)

# 针对每个列进行异常值替换
data = replace_outliers_with_median(data, 'Close_sf')
data = replace_outliers_with_median(data, 'Close_sm')
data = replace_outliers_with_median(data, 'Close_ss')

# 删除所有含有缺失值的行
data.dropna(inplace=True)

# 获取收盘价的列名
columns = data.columns[1:18:2]