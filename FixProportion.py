#coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import math

'''
    基于下车比例固定假设的对比算法
    输入：30号之前的日期
    输出：实时人数数组
'''


# 获取每小时内的平均流量（按15s计算）
def get_avg_flow():
    wt = pd.DataFrame(columns=['in_seconds'])
    dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
    , '20141208', '20141209', '20141210', '20141211', '20141212'
    , '20141215', '20141216', '20141217', '20141218', '20141219'
    , '20141223', '20141225', '20141226']
    for dt in dts:
        temp = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\WaitTime\waitTime_for14_line1_' + dt + '.csv')
        temp = temp.loc[:, ['in_seconds']]
        wt = wt.append(temp, ignore_index=True)
    x = []
    for i in range(1, 18):
        check_seconds = i * 3600
        flow = wt[(wt.in_seconds > (check_seconds - 3600)) & (wt.in_seconds <= check_seconds)].shape[0]
        avg_flow = float(flow) / (240*len(dts))
        x.append(avg_flow)
    return x


# 获得下车比例
def get_alight_proportion():
    star_time = datetime.datetime.strptime('20141201' + '063000', '%Y%m%d%H%M%S')
    end_time = datetime.datetime.strptime('20141228' + '233000', '%Y%m%d%H%M%S')
    transections = pd.DataFrame(columns=['in_station', 'out_station'])
    dts = ['201412_01_07_line_1_1276913', '201412_08_14_line_1', '201412_15_21_line_1', '201412_22_31_line_1']
    for dt in dts:
        temp = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\Transactions_' + dt + '.csv')
        temp = temp[(pd.to_datetime(temp.in_time) >= star_time) & (pd.to_datetime(temp.in_time) <= end_time)
                    & (temp.in_station > 14) & (temp.out_station <= 14)].loc[:, ['in_station', 'out_station']]
        transections = transections.append(temp, ignore_index=True)
    fenzi = transections[transections.out_station == 14].shape[0]
    fenmu = transections.shape[0]
    return float(fenzi)/fenmu


# 获得对比算法的预测结果
def get_fix_proportion_predict(test_dt='20141229'):
    C = 1880
    alight_proportion = get_alight_proportion()
    star_time = datetime.datetime.strptime(test_dt + '063000', '%Y%m%d%H%M%S')
    end_time = datetime.datetime.strptime(test_dt + '233000', '%Y%m%d%H%M%S')
    alight_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_22_31_line_1.csv')
    alight_df = alight_df[(alight_df.in_station > 14) & (alight_df.out_station == 14)
                          & (pd.to_datetime(alight_df.in_time) >= star_time)
                          & (pd.to_datetime(alight_df.in_time) <= end_time)].loc[:, ['in_time', 'total_time']]
    out_seconds = []
    for i in alight_df.index:
        out_seconds.append((datetime.datetime.strptime(alight_df.loc[i, 'in_time'], '%Y-%m-%d %H:%M:%S') - star_time).seconds + alight_df.loc[i, 'total_time'])
    alight_df['out_seconds'] = out_seconds
    real_time_table = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + test_dt + '.csv').loc[:, 'leav_time']
    real_time_table *= 15
    flow = get_avg_flow()
    walk_dua = 41  # 步行流量延迟
    predict_rs = []  # 存放预测的结果
    last_leav_sec = 0  # 上一班车的离开时间
    last_board = 0  # 上一班车的上车人数
    last_most = 0  # 上一班车的最高人数
    arr = 0  # 当前周期累计到达人数
    for i in range(204):
        leav_sec = real_time_table.loc[i]   # 下一班车离开时间
        for j in range((leav_sec - last_leav_sec)/15):
            if j == 0:
                predict_rs.append(last_most - last_board)   # 每个周期第一个时间片，等于剩余人数
            else:
                arr += flow[(last_leav_sec + (j + 1) * 15)/3600]
                predict_rs.append(round(last_most - last_board + arr))
        alight_num = alight_df[(alight_df.out_seconds > (last_leav_sec+walk_dua)) & (alight_df.out_seconds <= (leav_sec+walk_dua))].shape[0]
        on_train = min(C, float(alight_num)/alight_proportion)
        empty_num = min(C, round(C - on_train + alight_num))
        board_num = min(empty_num, predict_rs[-1])
        # print "alight: %d  ontrain: %d  empty: %d  board: %d" % (alight_num, on_train, empty_num, board_num)
        last_leav_sec = leav_sec
        last_board = board_num
        last_most = predict_rs[-1]
        arr = 0
    for i in range(4080-len(predict_rs)):
        predict_rs.append(last_most - last_board)
    return predict_rs

if __name__ == '__main__':
    test_dt = '20141229'
    test = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv').loc[
           :, ['check_seconds', 'count']]
    ha = get_fix_proportion_predict()
    plt.plot(test.check_seconds[0:800], test['count'][0:800], c='b')
    plt.plot(test.check_seconds[0:800], ha[0:800], c='r')
    plt.show()
    print r2_score(test['count'], ha)
    print mean_squared_error(test['count'], ha)
    print math.sqrt(mean_squared_error(test['count'], ha))
    print mean_absolute_error(test['count'], ha)