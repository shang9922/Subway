# coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
import ShortTermPredict
import seaborn as sns


def totalNum_And_boardNum(train_dt):
    print('Combining data ...')
    train_df = pd.DataFrame(
        columns=["train_num", "time_pice", "day_type", "duration", "leav_time", "arr_num", "total_num",
                 "board_num", "left_num", "pre_1", "pre_2", "pre_3"])
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)
    most_totalNum = max(train_df.total_num)
    lest_totalNum = min(train_df.total_num)

    step = 10

    terms = most_totalNum / step
    print terms
    x = []
    y = []
    for i in range(terms):
        a_limit = i * step
        b_limit = (i + 1) * step
        x.append(i + 1)
        cut = train_df[(train_df.total_num >= a_limit) & (train_df.total_num <= b_limit)].loc[:, 'board_num']
        if cut.shape[0] != 0:
            y.append(np.mean(cut.values))
        else:
            y.append(0)
    # plt.bar(x, y)
    # plt.show()
    print y


def trainNum_And_boardNum(train_dt):
    print('Combining data ...')
    train_df = pd.DataFrame(
        columns=["train_num", "time_pice", "day_type", "duration", "leav_time", "arr_num", "total_num",
                 "board_num", "left_num", "pre_1", "pre_2", "pre_3"])
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)

    x = []
    y = []
    for i in range(1, 205):
        x.append(i)
        cut = train_df[train_df.train_num == i].loc[:, 'board_num']
        if cut.shape[0] != 0:
            y.append(np.mean(cut.values))
        else:
            y.append(0)
    plt.bar(x, y)
    plt.show()
    # print y


def duration_And_boardNum(train_dt):
    print('Combining data ...')
    train_df = pd.DataFrame(
        columns=["train_num", "time_pice", "day_type", "duration", "leav_time", "arr_num", "total_num",
                 "board_num", "left_num", "pre_1", "pre_2", "pre_3"])
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)
    most_duration = max(train_df.duration)
    lest_duration = min(train_df.duration)

    step = 1

    terms = most_duration / step

    x = []
    y = []
    for i in range(terms + 1):
        a_limit = i * step
        b_limit = (i + 1) * step
        x.append(i + 1)
        cut = train_df[(train_df.duration >= a_limit) & (train_df.duration <= b_limit)].loc[:, 'board_num']
        if cut.shape[0] != 0:
            y.append(np.mean(cut.values))
        else:
            y.append(0)
    plt.bar(x, y)
    plt.show()
    # print y


def time_pice_And_boardNum(train_dt):
    print('Combining data ...')
    train_df = pd.DataFrame(
        columns=["train_num", "time_pice", "day_type", "duration", "leav_time", "arr_num", "total_num",
                 "board_num", "left_num", "pre_1", "pre_2", "pre_3"])
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)
    most_time_pice = max(train_df.time_pice)
    lest_time_pice = min(train_df.time_pice)

    step = 1

    terms = most_time_pice / step

    x = []
    y = []
    for i in range(1, terms + 1):
        x.append(i)
        cut = train_df[train_df.time_pice == i].loc[:, 'board_num']
        if cut.shape[0] != 0:
            y.append(np.mean(cut.values))
        else:
            y.append(0)
    plt.bar(x, y)
    plt.show()
    # print y


def inFlow(train_dt):
    print('Combining data ...')
    train_df = pd.DataFrame(
        columns=["check_seconds", "flow"])
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)
    step = 204
    x = []
    y = []

    for i in range(step + 1):
        checkSeconds = i * 300
        x.append((datetime.datetime.strptime('20150101063000', '%Y%m%d%H%M%S') + datetime.timedelta(
            seconds=int(checkSeconds))).time())
        cut = train_df[train_df.check_seconds == checkSeconds].loc[:, 'flow']
        if cut.shape[0] != 0:
            y.append(np.mean(cut.values) * 22)
        else:
            y.append(0)
    plt.plot(x, y)
    plt.title('Passenger Flow and Check-in Time')
    plt.xlabel('Time')
    plt.ylabel('Passenger Flow')
    plt.xticks(['07:30', '10:00', '17:00', '19:30', '23:30'])
    plt.show()


def inFlow_whether():
    train_df_sun = pd.DataFrame(
        columns=["check_seconds", "flow"])
    train_df_rain = pd.DataFrame(
        columns=["check_seconds", "flow"])
    temp_df = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_20141203.csv')
    train_df_sun = train_df_sun.append(temp_df, ignore_index=True)
    temp_df = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_20141201.csv')
    train_df_rain = train_df_rain.append(temp_df, ignore_index=True)
    step = 34
    x = []
    y_sun = []
    y_rain = []

    for i in range(step + 1):
        checkSeconds = i * 1800
        x.append((datetime.datetime.strptime('20150101063000', '%Y%m%d%H%M%S') + datetime.timedelta(
            seconds=int(checkSeconds))).time())
        cut = train_df_sun[train_df_sun.check_seconds == checkSeconds].loc[:, 'flow']
        if cut.shape[0] != 0:
            y_sun.append(np.mean(cut.values) * 22)
        else:
            y_sun.append(0)
        cut = train_df_rain[train_df_rain.check_seconds == checkSeconds].loc[:, 'flow']
        if cut.shape[0] != 0:
            y_rain.append(np.mean(cut.values) * 22)
        else:
            y_rain.append(0)

    plt.plot(x, y_sun, lw=2, marker='s', label='2015-04-29 Sunny')
    plt.plot(x, y_rain, c='r', lw=2, marker='^', label='2015-04-30 Rainy')
    plt.title('Passenger Flow and Weather')
    plt.xlabel('Time')
    plt.ylabel('Passenger Flow')
    plt.xticks(['06:30', '08:30', '10:30', '12:30', '14:30', '16:30', '18:30', '20:30', '22:30'])
    plt.legend()
    plt.show()


def inFlow_duration(train_dt):
    train_df_1min = pd.DataFrame(
        columns=["check_seconds", "flow"])
    train_df_5min = pd.DataFrame(
        columns=["check_seconds", "flow"])
    train_df_30min = pd.DataFrame(
        columns=["check_seconds", "flow"])
    train_df_60min = pd.DataFrame(
        columns=["check_seconds", "flow"])
    wt = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\WaitTime\waitTime_for14_line1_' + train_dt + '.csv')
    step1 = 240
    step5 = 48
    step30 = 8
    step60 = 4
    dura1 = 60
    dura5 = 300
    dura30 = 1800
    dura60 = 3600
    x1 = []
    x5 = []
    x30 = []
    x60 = []
    y1 = []
    y5 = []
    y30 = []
    y60 = []
    for i in range(step1 + 1):
        check_seconds = i * dura1
        x1.append((datetime.datetime.strptime('20150101063000', '%Y%m%d%H%M%S') + datetime.timedelta(
            seconds=int(check_seconds))).time())
        flow = wt[(wt.in_seconds > (check_seconds - dura1)) & (wt.in_seconds <= check_seconds)].shape[0]
        y1.append(flow * 5)
    for i in range(step5 + 1):
        check_seconds = i * dura5
        x5.append((datetime.datetime.strptime('20150101063000', '%Y%m%d%H%M%S') + datetime.timedelta(
            seconds=int(check_seconds))).time())
        flow = wt[(wt.in_seconds > (check_seconds - dura5)) & (wt.in_seconds <= check_seconds)].shape[0]
        y5.append(flow)
    for i in range(step30 + 1):
        check_seconds = i * dura30
        x30.append((datetime.datetime.strptime('20150101063000', '%Y%m%d%H%M%S') + datetime.timedelta(
            seconds=int(check_seconds))).time())
        flow = wt[(wt.in_seconds > (check_seconds - dura30)) & (wt.in_seconds <= check_seconds)].shape[0]
        y30.append(flow / 6)
    for i in range(step60 + 1):
        check_seconds = i * dura60
        x60.append((datetime.datetime.strptime('20150101063000', '%Y%m%d%H%M%S') + datetime.timedelta(
            seconds=int(check_seconds))).time())
        flow = wt[(wt.in_seconds > (check_seconds - dura60)) & (wt.in_seconds <= check_seconds)].shape[0]
        y60.append(flow / 12)
    plt.plot(x1, y1, lw=2, c='y', label='1min')
    plt.plot(x5, y5, lw=2, marker='^', label='5min')
    plt.plot(x30, y30, lw=2, c='r', marker='o', label='30min')
    plt.plot(x60, y60, lw=2, c='k', marker='s', label='60min')
    plt.title('Passenger Flow on 2015-05-08')
    plt.xlabel('Time')
    plt.ylabel('Passenger Flow')
    plt.xticks(['06:30', '07:30', '08:30', '09:30', '10:30'])
    plt.legend()
    plt.show()


def inFlow_mutidays():
    train_df = pd.DataFrame(columns=['check_seconds', 'flow'])
    print('Combining data ...')
    train_dt = ['20141201', '20141202', '20141203', '20141204', '20141205'
        , '20141208', '20141209', '20141210', '20141211', '20141212'
        , '20141215', '20141216', '20141217', '20141218', '20141219']
    x = []
    y = []
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)
        # for i in range(temp_df.shape[0]):
        #     x.append((datetime.datetime.strptime(date_str + '063000', '%Y%m%d%H%M%S') + datetime.timedelta(
        #     seconds=int(temp_df.loc[i,['check_seconds']]))))
        #     y.append(temp_df.loc[i,['flow']])
    len = train_df.shape[0]
    train_df = train_df.loc[:, ['flow']]
    train_df = pd.DataFrame(train_df.flow.values, index=pd.date_range('2008-01-01', periods=len), columns=['flow'])
    train_df.index = pd.to_datetime(train_df.index)
    plt.plot(train_df.index, train_df['flow'])
    plt.title('Passenger Flow and Date')
    plt.xlabel('Date')
    plt.ylabel('Passenger Flow')
    # plt.xticks(['07:30', '10:00', '17:00', '19:30', '23:30'])
    plt.show()


def inFlow_FenJie():
    train_df = pd.DataFrame(columns=['check_seconds', 'flow'])
    print('Combining data ...')
    train_dt = ['20141201', '20141202', '20141203', '20141204'
        , '20141208', '20141209', '20141210', '20141211'
        , '20141215', '20141216', '20141217', '20141218', '20141219', '20141223', '20141225']
    x = []
    y = []
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)
    len = train_df.shape[0]
    train_df = train_df.loc[:, ['flow']]
    train_df = pd.DataFrame(train_df.flow.values, index=pd.date_range('2008-01-01', periods=len), columns=['flow'])
    train_df.index = pd.to_datetime(train_df.index)
    train_df['flow'] = train_df['flow'].astype(float)
    ts = train_df['flow']
    decomposition = seasonal_decompose(ts, freq=204, two_sided=False, model='additive')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    seasonal.plot()
    plt.xlabel('Date')
    plt.ylabel('Cycle')
    plt.show()


def tongji_transections():
    file_name = ['E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_01_07_line_1_1276913.csv',
                 'E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_08_14_line_1.csv',
                 'E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_15_21_line_1.csv',
                 'E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_22_31_line_1.csv']
    count = 0
    for i in file_name:
        count += pd.read_csv(i).shape[0]
    count = count * 2 * 0.8 * 5 / 7
    print count


def choose_K():
    train_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\mean_squared_error.csv')
    x = []
    y2 = []
    y = []
    dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
        , '20141208', '20141209', '20141210', '20141211', '20141212'
        , '20141215', '20141216', '20141217', '20141218', '20141219'
        , '20141223', '20141225', '20141226']
    temp = ShortTermPredict.get_HA_mean_squared_error(dts)
    for i in range(train_df.shape[0]):
        x.append(i + 1)
        y.append(train_df.iloc[i, 1])
        y2.append(temp)
    plt.plot(x, y, label='k-Steps')
    plt.plot(x, y2, c='r', ls='--', label='HA')
    plt.title('Mean Squared Error and k')
    plt.xlabel('k')
    plt.ylabel('MSE')
    plt.xticks(x)
    plt.yticks([3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750])
    plt.legend()
    plt.show()


def p_and_q_ReLiTu():
    df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\Bic_matrix_26.csv', usecols=[1, 2, 3, 4, 5, 6, 7])
    data = df.values
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    f, ax = plt.subplots(figsize=(10, 4))
    cmap = sns.cubehelix_palette(start=1.5, rot=3, gamma=0.8, as_cmap=True)
    sns.heatmap(data, annot=True, fmt='.0f', mask=(data > -9000), ax=ax, vmax=-12000, vmin=-12091, center=-12050)
    ax.set_title('Heatmap of BIC')
    ax.set_xlabel('Value of q', font1)
    ax.set_ylabel('Value of p', font1)
    plt.show()


if __name__ == '__main__':
    # train_dt = ['20141201', '20141202', '20141203', '20141204', '20141205', '20141206', '20141207'
    #     , '20141208', '20141209', '20141210', '20141211', '20141212', '20141213', '20141214'
    #     , '20141215', '20141216', '20141217', '20141218', '20141219', '20141220', '20141221'
    #     , '20141223', '20141224', '20141225', '20141226', '20141227', '20141228', '20141229']
    train_dt = ['20141201', '20141202', '20141203', '20141204', '20141205'
        , '20141208', '20141209', '20141210', '20141211', '20141212'
        , '20141215', '20141216', '20141217', '20141218', '20141219'
        , '20141223', '20141225', '20141226']
    # train_dt = ['20141229']
    # totalNum_And_boardNum(train_dt)
    # trainNum_And_boardNum(train_dt)
    # duration_And_boardNum(train_dt)
    # time_pice_And_boardNum(train_dt)
    # inFlow(train_dt)
    # inFlow_whether()
    # inFlow_duration('20141229')
    # inFlow_mutidays()
    # inFlow_FenJie()
    # tongji_transections()
    # choose_K()
    # p_and_q_ReLiTu()
