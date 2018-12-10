# coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

'''补充特征，获得最终的训练数据集'''


def get_train_date_by_date(dt):
    tt = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\TimeTable\TimeTable_for14_line1_" + dt + ".csv", usecols=[1])
    wt = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\WaitTime\waitTime_for14_line1_" + dt + ".csv")

    for j in range(tt.shape[0]):  # 个别时刻没对准到15s整数倍，将其对整
        temp = tt.iloc[j, 0]
        yu = temp % 15
        if yu != 0:
            tt.iloc[j, 0] = temp - yu

    df = pd.DataFrame(
        columns=["train_num", "time_pice", "day_type", "duration", "leav_time", "arr_num", "total_num", "board_num", "left_num", "pre_1", "pre_2", "pre_3"])

    for i in range(tt.shape[0]):
        train_num = i + 1
        df.loc[i, 'train_num'] = train_num   # 车次
        df.iloc[i, 1] = get_time_pice(tt.loc[i, 'check_seconds'])  # 时间段

        if train_num == 1:                                         # 距离上一班车的时间片数
            df.iloc[i, 3] = tt.loc[i, 'check_seconds']/15
        else:
            df.iloc[i, 3] = (tt.loc[i, 'check_seconds'] - tt.loc[i - 1, 'check_seconds'])/15

        df.iloc[i, 4] = tt.loc[i, 'check_seconds']/15      # 发车时间

        leav_seconds = tt.loc[i, 'check_seconds']
        df.iloc[i, 6] = wt[(wt.arr_seconds <= leav_seconds) & (wt.leav_seconds > leav_seconds)].shape[0]  # 站台总人数

        check_seconds = leav_seconds + 15
        if check_seconds >= 61200:                        # 遗留人数
            df.iloc[i, 8] = 0
        else:
            df.iloc[i, 8] = wt[(wt.arr_seconds <= check_seconds) & (wt.leav_seconds > check_seconds)].shape[0]

        if train_num == 1:                                         # 累计到达人数
            df.iloc[i, 5] = df.iloc[i, 6]
        else:
            df.iloc[i, 5] = df.iloc[i, 6] - df.iloc[i - 1, 8]

        df.iloc[i, 7] = df.iloc[i, 6] - df.iloc[i, 8]       # 上车人数

        if i == 0:
            df.iloc[i, 9] = 0
            df.iloc[i, 10] = 0
            df.iloc[i, 11] = 0
        elif i == 1:
            df.iloc[i, 10] = 0
            df.iloc[i, 11] = 0
            df.iloc[i, 9] = df.iloc[i - 1, 7]
        elif i == 2:
            df.iloc[i, 11] = 0
            df.iloc[i, 9] = df.iloc[i - 1, 7]
            df.iloc[i, 10] = df.iloc[i - 2, 7]
        else:
            df.iloc[i, 9] = df.iloc[i - 1, 7]
            df.iloc[i, 10] = df.iloc[i - 2, 7]
            df.iloc[i, 11] = df.iloc[i - 3, 7]

    day_of_dt = datetime.datetime.strptime(dt, '%Y%m%d')
    week_day = day_of_dt.weekday()  # 获得日期对应星期几，并决定day_type
    if week_day > 4:
        df['day_type'] = 2  # 周末对应2
    else:
        df['day_type'] = 1  # 周一对应1
    df.to_csv("E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_" + dt + ".csv")


def get_time_pice(check_seconds):
    if 0 <= check_seconds < 1800:           # 早高峰前
        return 1
    elif 1800 <= check_seconds < 14400:      # 早高峰上升段
        return 2
    elif 14400 <= check_seconds < 37800:    # 中间段
        return 3
    elif 37800 <= check_seconds < 46800:    # 晚高峰上升段
        return 4
    elif 46800 <= check_seconds <= 61200:   # 晚高峰后
        return 5

dts = ['20141206', '20141207'
    , '20141208', '20141209', '20141210', '20141211', '20141212', '20141213', '20141214'
    , '20141215', '20141216', '20141217', '20141218', '20141219', '20141220', '20141221'
    , '20141223', '20141224', '20141225', '20141226', '20141227', '20141228'
    , '20141229']
for dt in dts:
    get_train_date_by_date(dt)
