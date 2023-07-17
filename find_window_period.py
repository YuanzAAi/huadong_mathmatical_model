# 定义一个函数，用来根据判据找出关系存在的窗口期
def find_window_period(result_list, threshold):
    # 初始化空列表，用来存储关系存在的窗口期
    window_period_list = []
    # 初始化标志变量，用来记录是否进入关系存在的状态
    flag = False
    # 初始化变量，用来记录关系存在的起始索引
    start_index = None
    # 循环遍历每个滑动窗口的回归结果
    for i, (params, rsquared, aic) in enumerate(result_list):
        # 计算一个综合的判据，根据决定系数判断
        criterion =  rsquared
        # 如果综合判据大于阈值，并且当前不在关系存在的状态中
        if criterion > threshold and not flag:
            # 将标志变量设为 True，表示进入关系存在的状态
            flag = True
            # 将当前索引设为关系存在的起始索引
            start_index = i
        # 如果综合判据小于等于阈值，并且当前在关系存在的状态中
        elif criterion <= threshold and flag:
            # 将标志变量设为 False，表示退出关系存在的状态
            flag = False
            # 将当前索引设为关系存在的结束索引，并与起始索引组成一个元组，添加到窗口期列表中
            end_index = i - 1
            window_period_list.append((start_index, end_index))
            # 将起始索引设为 None，表示重新寻找关系存在的起始点
            start_index = None

    # 如果最后还在关系存在的状态中，那么将最后一个索引设为关系存在的结束索引，并与起始索引组成一个元组，添加到窗口期列表中
    if flag:
        end_index = len(result_list) - 1
        window_period_list.append((start_index, end_index))

    # 返回窗口期列表
    return window_period_list