#coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import math

'''
    基于下车平均下车人数假设的对比算法
    输入：30号之前的日期
    输出：实时人数数组
'''


# 获取每半小时内的平均流量（按15s计算）
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
    for i in range(1, 35):
        check_seconds = i * 1800
        flow = wt[(wt.in_seconds > (check_seconds - 1800)) & (wt.in_seconds <= check_seconds)].shape[0]
        avg_flow = float(flow) / (120*len(dts))
        x.append(avg_flow)
    return x


# 获得按平均法获得上车比例
def get_board_proportion():
    td = pd.DataFrame(columns=['total_num', 'board_num'])
    dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
        , '20141208', '20141209', '20141210', '20141211', '20141212'
        , '20141215', '20141216', '20141217', '20141218', '20141219'
        , '20141223', '20141225', '20141226']
    trains = 0
    for dt in dts:
        temp = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + dt + '.csv').loc[:, ['total_num', 'board_num']]
        td = td.append(temp, ignore_index=True)
        trains += 204
    rs = float(sum(td.board_num))/sum(td.total_num)
    return rs



# 获得对比算法的预测结果
def get_fix_board_predict(test_dt='20141229'):
    C = 1880
    real_time_table = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + test_dt + '.csv').loc[:, 'leav_time']
    real_time_table *= 15
    flow = get_avg_flow()
    walk_dua = 41  # 步行流量延迟
    board_proportion = 0.85#get_board_proportion()
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
                arr += flow[(last_leav_sec + (j + 1) * 15)/1800]
                predict_rs.append(round(last_most - last_board + arr))
        board_num = min(C, round(predict_rs[-1]*board_proportion))
        # print "alight: %d  ontrain: %d  empty: %d  board: %d" % (alight_num, on_train, empty_num, board_num)
        last_leav_sec = leav_sec
        last_board = board_num
        last_most = predict_rs[-1]
        arr = 0
    for i in range(4080-len(predict_rs)):
        predict_rs.append(0)
    return predict_rs

if __name__ == '__main__':
    test_dt = '20141229'
    test = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv').loc[
           :, ['check_seconds', 'count']]
    ha = get_fix_board_predict()
    plt.plot(test.check_seconds[2400:3200], test['count'][2400:3200], c='b')
    plt.plot(test.check_seconds[2400:3200], ha[2400:3200], c='r')
    plt.show()
    print r2_score(test['count'], ha)
    print mean_squared_error(test['count'], ha)
    print math.sqrt(mean_squared_error(test['count'], ha))
    print mean_absolute_error(test['count'], ha)