# coding:utf-8
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import ShortTermPredict as STP
import matplotlib.pyplot as plt
import math
import ArimaPredict as AP
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# 根据日期字符串获取当天实时人数数据
def get_platform_count(dt, out_put=False):
    df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\WaitTime\waitTime_for14_line1_' + dt + '.csv')
    x = []
    y = []
    for i in range(1, 4081):
        check_seconds = i * 15
        x.append(check_seconds)
        count = df[(df.arr_seconds <= check_seconds) & (df.leav_seconds > check_seconds)].shape[0]
        y.append(count)
    # plt.plot(x, y)
    # plt.show()
    rs_df = pd.DataFrame({'check_seconds': x,
                          'count': y},columns=['check_seconds', 'count'])
    if out_put:
        rs_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + dt + '.csv')
    return rs_df


# 获取对应时间片
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


# 将ARIMA预测的流量拆解为15s时间片的流量
def decompose_flow(Arima_rs):
    flow_rs = []
    for long_flow in Arima_rs:
        short_flow = float(long_flow)/20    # 这里取整会丢失精度，应该在计算完累计到达人数后再取整
        for i in range(20):
            flow_rs.append(short_flow)
    return flow_rs


# 历史平均预测
def HA_predict():
    dts = ['20141201', '20141202', '20141203', '20141204', '20141205', '20141206', '20141207'
        , '20141208', '20141209', '20141210', '20141211', '20141212', '20141213', '20141214'
        , '20141215', '20141216', '20141217', '20141218', '20141219', '20141220', '20141221'
        , '20141223', '20141225', '20141226', '20141227', '20141228']
    datas = pd.DataFrame(columns=['check_seconds', 'count'])
    X = []
    Y = []
    for dt in dts:
        temp_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + dt + '.csv').loc[:, ['check_seconds', 'count']]
        temp_df.columns = ['check_seconds', 'count']
        datas = datas.append(temp_df, ignore_index=True)
    for i in range(1, 4081):
        sec = i * 15
        X.append(sec)
        avg_count = round(np.mean(datas[datas.check_seconds == sec]['count']))
        Y.append(avg_count)
    ha_df = pd.DataFrame({'check_seconds': X,
                          'count': Y}, columns=['check_seconds', 'count'])
    return ha_df


# 获得SVR训练数据
def get_svr_train(train_dt):
    train_df = pd.DataFrame(
        columns=["train_num", "time_pice", "day_type", "duration", "leav_time", "arr_num", "total_num",
                 "board_num", "left_num", "pre_1", "pre_2", "pre_3"])

    print('SVR Combining data ...')
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)
    X_train = pd.DataFrame({'train_num': train_df.train_num,
                            'time_pice': train_df.time_pice,
                            'day_type': train_df.day_type,
                            'duration': train_df.duration,
                            'leav_time': train_df.leav_time,
                            'total_num': train_df.total_num,
                            'pre_1': train_df.pre_1,
                            'pre_2': train_df.pre_2,
                            'pre_3': train_df.pre_3},
                           columns=['train_num', 'time_pice', 'day_type', 'duration', 'leav_time', 'total_num',
                                    'pre_1', 'pre_2', 'pre_3']).values
    Y_train = train_df.board_num.values
    rs = []
    rs.append(X_train)
    rs.append(Y_train)
    return rs


# 训练SVR
def get_SVR_model(train_data):
    X_train = train_data[0]
    Y_train = train_data[1]
    ss_X = StandardScaler()  # 训练数据和测试数据的放缩
    ss_y = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    Y_train = ss_y.fit_transform(Y_train.reshape(-1, 1))
    linear_svr = SVR(kernel='linear')  # 线性核函数初始化的SVR
    linear_svr.fit(X_train, Y_train)
    return linear_svr


# 用SVR预测
def predict_Board_num(train_data, svr_model, X_test):
    X_train = train_data[0]
    Y_train = train_data[1]
    ss_X = StandardScaler()  # 训练数据和测试数据的放缩
    ss_y = StandardScaler()
    ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    ss_y.fit_transform(Y_train.reshape(-1, 1))
    Y_predict = svr_model.predict(X_test)
    Y_predict = ss_y.inverse_transform(Y_predict)
    return Y_predict


# 基于真实时刻表预测
def real_time_table_predict(dts, svr_dt, test_dt='20141229'):
    avg_time_table = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_'+ test_dt + '.csv').loc[:, 'leav_time']
    avg_time_table *= 15
    Arima_rs = AP.arimaMainProcess(dts, 204, test_dt)
    flow = decompose_flow(Arima_rs)     # 存放预测得到的流量，以15s为单位，没有进过取整处理
    svr_train_data = get_svr_train(svr_dt)
    svr_model = get_SVR_model(svr_train_data)    # 训练好的SVR模型
    walk_dua = 3            # 步行流量延迟
    predict_rs = []         # 存放预测的结果
    last_leav_sec = 0       # 上一班车的离开时间
    last_board = 0          # 上一班车的上车人数
    last_most = 0           # 上一班车的最高人数
    arr = 0                 # 当前周期累计到达人数
    pre_1 = 0
    pre_2 = 0
    pre_3 = 0
    for i in range(204):
        leav_sec = avg_time_table.loc[i]   # 下一班车离开时间
        for j in range((leav_sec - last_leav_sec)/15):
            if j == 0:
                predict_rs.append(last_most - last_board)   # 每个周期第一个时间片，等于剩余人数
            else:
                if len(predict_rs) >= walk_dua:
                    arr += flow[len(predict_rs) - walk_dua]
                predict_rs.append(round(last_most - last_board + arr))

        # 构造训练特征
        day_of_dt = datetime.datetime.strptime(test_dt, '%Y%m%d')
        week_day = day_of_dt.weekday()  # 获得日期对应星期几，并决定day_type
        daytype = 0
        if week_day > 4:
            daytype = 2  # 周末对应2
        else:
            daytype = 1  # 周一对应1
        X_test = pd.DataFrame({'train_num': [i + 1],
                            'time_pice': [get_time_pice(leav_sec)],
                            'day_type': [daytype],
                            'duration': [(leav_sec - last_leav_sec)/15],
                            'leav_time': [leav_sec/15],
                            'total_num': [predict_rs[-1]],
                            'pre_1': [pre_1],
                            'pre_2': [pre_2],
                            'pre_3': [pre_3]},
                           columns=['train_num', 'time_pice', 'day_type', 'duration', 'leav_time', 'total_num',
                                    'pre_1', 'pre_2', 'pre_3']).values
        Y_pre = predict_Board_num(svr_train_data, svr_model, X_test)
        board_num = math.ceil(Y_pre)
        if i == 203:                        # 最后一趟车装走所有乘客
            board_num = predict_rs[-1]
        last_leav_sec = leav_sec
        last_board = board_num
        last_most = predict_rs[-1]
        arr = 0
        pre_1 = pre_2
        pre_2 = pre_3
        pre_3 = board_num
    for i in range(4080-len(predict_rs)):
        predict_rs.append(0)
    return predict_rs


'''每轮预测截止时间片为真实时刻表中对应最后一班车的发车时间片+1'''
# 基于短时时刻表的预测
def short_term_predict(dts, svr_dt, test_dt='20141229', steps = 3):
    # 预测时刻表的过程，步长并不是越短越好，时刻表预测步长固定为3
    predict_time_table = STP.predict_by_constant_weights(dts, test_dt, 3)  # 获得预测的时刻表，从第5班车开始，长度200
    Arima_rs = AP.arimaMainProcess(dts, 204, test_dt)
    flow = decompose_flow(Arima_rs)     # 存放预测得到的流量，以15s为单位，没有进过取整处理
    svr_train_data = get_svr_train(svr_dt)
    svr_model = get_SVR_model(svr_train_data)    # 训练好的SVR模型
    walk_dua = 3            # 步行流量延迟
    predict_rs = []         # 存放预测的结果

    real_train_data = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + test_dt + '.csv')
    real_time_table = real_train_data.loc[:, 'leav_time'].copy()       # 真实的时刻表
    real_time_table *= 15

    # 小改
    # for i in range(len(predict_time_table)):
    #     re = real_time_table.loc[4+i]
    #     pre = predict_time_table[i]
    #     if abs(re-pre) > 30:
    #         if re > pre:
    #             predict_time_table[i] = re - 30
    #         else:
    #             predict_time_table[i] = re + 30

    # 真实的实时人数
    real_count = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv').loc[:, ['check_seconds', 'count']]

    term_num = math.ceil(float(len(predict_time_table))/steps)  # 预测的轮数
    last_remain_sec = real_time_table.loc[3] + 15        # 上一班车出发时间延后一个时间片的时间
    last_remain_num = real_count[real_count.check_seconds == last_remain_sec].loc[:, 'count']
    pre_1 = real_train_data.loc[4, 'pre_1']
    pre_2 = real_train_data.loc[4, 'pre_2']
    pre_3 = real_train_data.loc[4, 'pre_3']
    pre_tab = []
    pre_rem = []
    for i in range(int(term_num)):                       # 每轮
        last_most = 0           # 前一班车的最高人数
        last_board = 0          # 前一班车的上车人数
        arr = 0                 # 当前周期累计到达人数
        del pre_tab[:]
        del pre_rem[:]
        most_train = min((i + 1) * steps + 4, real_time_table.shape[0])     # 本轮预测到第几班车
        pice_num = (real_time_table[most_train - 1] - last_remain_sec)/15 + 1   # 本轮总共预测的时间片数
        for p in range(i * steps, most_train - 4):
            pre_tab.append(predict_time_table[p])           # 本轮所有车的预测发车时刻
            pre_rem.append(predict_time_table[p] + 15)      # 本轮所有车的预测遗留时刻
        for j in range(pice_num):
            ind = i * steps + 3     # 最近一个真实时刻的车次下标
            now_sec = real_time_table[ind] + 15 + 15*(j+1)
            if now_sec in pre_tab:      # 发车时刻
                arr += flow[real_time_table.loc[ind]/15 + 1 + j - walk_dua]
                predict_rs.append(round(last_remain_num + arr))
                last_most = predict_rs[-1]
                # 构造预测特征
                day_of_dt = datetime.datetime.strptime(test_dt, '%Y%m%d')
                week_day = day_of_dt.weekday()  # 获得日期对应星期几，并决定day_type
                daytype = 0
                if week_day > 4:
                    daytype = 2  # 周末对应2
                else:
                    daytype = 1  # 周一对应1
                X_test = pd.DataFrame({'train_num': [ind + 1 + 4],
                                       'time_pice': [get_time_pice(now_sec)],
                                       'day_type': [daytype],
                                       'duration': [(now_sec + 15 - last_remain_sec) / 15],
                                       'leav_time': [now_sec / 15],
                                       'total_num': [last_most],
                                       'pre_1': [pre_1],
                                       'pre_2': [pre_2],
                                       'pre_3': [pre_3]},
                                      columns=['train_num', 'time_pice', 'day_type', 'duration', 'leav_time',
                                               'total_num',
                                               'pre_1', 'pre_2', 'pre_3']).values
                Y_pre = predict_Board_num(svr_train_data, svr_model, X_test)
                last_board = min(math.ceil(Y_pre), last_most)
                # 更新临时变量
                last_remain_sec = now_sec + 15
                last_remain_num = last_most - last_board
                arr = 0
                pre_1 = pre_2
                pre_2 = pre_3
                pre_3 = last_board
            elif now_sec in pre_rem:        # 发车下一个时刻
                predict_rs.append(last_most - last_board)
            else:       # 普通时间片
                arr += flow[real_time_table.loc[ind]/15 + 1 + j - walk_dua]
                predict_rs.append(round(last_remain_num + arr))

        last_remain_sec = real_time_table.loc[most_train - 1] + 15  # 上一班车出发时间延后一个时间片的时间
        last_remain_num = real_count[real_count.check_seconds == last_remain_sec].loc[:, 'count']
        if most_train == real_time_table.shape[0]:
            pre_1 = real_train_data.loc[most_train-1, 'board_num']
            pre_2 = real_train_data.loc[most_train-1, 'pre_1']
            pre_3 = real_train_data.loc[most_train-1, 'pre_2']
        else:
            pre_1 = real_train_data.loc[most_train, 'pre_1']
            pre_2 = real_train_data.loc[most_train, 'pre_2']
            pre_3 = real_train_data.loc[most_train, 'pre_3']
    l = len(predict_rs)
    for i in range(4080 - l - real_train_data.loc[3, 'leav_time'] - 1):
        predict_rs.append(0)
    return predict_rs

# test_dt = '20141229'
# dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
#     , '20141208', '20141209', '20141210', '20141211', '20141212'
#     , '20141215', '20141216', '20141217', '20141218', '20141219'
#     , '20141223', '20141225', '20141226']
# svr_dt = ['20141201', '20141202', '20141203', '20141204', '20141205', '20141206', '20141207'
#     , '20141208', '20141209', '20141210', '20141211', '20141212', '20141213', '20141214'
#     , '20141215', '20141216', '20141217', '20141218', '20141219', '20141220', '20141221'
#     , '20141223', '20141224', '20141225', '20141226', '20141227', '20141228']
# weekend_dts = ['20141206', '20141207', '20141213', '20141214', '20141220', '20141221'
#     , '20141227', '20141228']

# Arima_rs = AP.arimaMainProcess(dts, 204, test_dt)
# flow = decompose_flow(Arima_rs)

# x = [i for i in range(4080)]
# plt.plot(x, flow)
# plt.show()

'''历史平均预测'''
# test = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv').loc[:, ['check_seconds', 'count']]
# # ha = HA_predict(dts).loc[:, 'count']
# ha = real_time_table_predict(dts, test_dt)
# plt.plot(test.check_seconds[200:800], test['count'][200:800], c='b')
# plt.plot(test.check_seconds[200:800], ha[200:800], c='r')
# plt.show()
# print r2_score(test['count'], ha)
# print mean_squared_error(test['count'], ha)
# print math.sqrt(mean_squared_error(test['count'], ha))
# print mean_absolute_error(test['count'], ha)


# 进行短期预测
# predict_rs = short_term_predict(dts, svr_dt, test_dt, steps=3)
# real_train_data = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + test_dt + '.csv')
# real_count = pd.read_csv(
#         'E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv').loc[:, ['count']]
# test = pd.read_csv(
#         'E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv')
# start_index = real_train_data.loc[3, 'leav_time'] + 1
# real_data = real_count.loc[start_index:, ['count']]
# print r2_score(real_data['count'], predict_rs)
# print mean_squared_error(real_data['count'], predict_rs)
# print math.sqrt(mean_squared_error(real_data['count'], predict_rs))
# print mean_absolute_error(real_data['count'], predict_rs)
# plt.plot(test.check_seconds[0:600], real_data['count'][0:600], c='b', label='real data')
# plt.plot(test.check_seconds[0:600], predict_rs[0:600], c='r', label='predict value')
# plt.legend()
# plt.show()
# plt.plot(test.check_seconds[600:1200], real_data['count'][600:1200], c='b')
# plt.plot(test.check_seconds[600:1200], predict_rs[600:1200], c='r')
# plt.show()
# plt.plot(test.check_seconds[1200:1800], real_data['count'][1200:1800], c='b')
# plt.plot(test.check_seconds[1200:1800], predict_rs[1200:1800], c='r')
# plt.show()
# plt.plot(test.check_seconds[1800:2400], real_data['count'][1800:2400], c='b')
# plt.plot(test.check_seconds[1800:2400], predict_rs[1800:2400], c='r')
# plt.show()
