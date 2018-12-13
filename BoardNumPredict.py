# coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


def mianProcess(train_dt, test_dt):
    train_df = pd.DataFrame(
        columns=["train_num", "time_pice", "day_type", "duration", "leav_time", "arr_num", "total_num",
                 "board_num", "left_num", "pre_1", "pre_2", "pre_3"])

    print('Combining data ...')
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)
    test_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + test_dt + '.csv')

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

    X_test = pd.DataFrame({'train_num': test_df.train_num,
                           'time_pice': test_df.time_pice,
                           'day_type': test_df.day_type,
                           'duration': test_df.duration,
                           'leav_time': test_df.leav_time,
                           'total_num': test_df.total_num,
                           'pre_1': test_df.pre_1,
                            'pre_2': test_df.pre_2,
                            'pre_3': test_df.pre_3},
                           columns=['train_num', 'time_pice', 'day_type', 'duration', 'leav_time', 'total_num',
                                    'pre_1', 'pre_2', 'pre_3']).values
    Y_test = test_df.board_num.values

    ss_X = StandardScaler()     # 训练数据和测试数据的放缩
    ss_y = StandardScaler()
    X_train = ss_X.fit_transform(X_train)
    X_test = ss_X.transform(X_test)
    Y_train = ss_y.fit_transform(Y_train.reshape(-1, 1))
    Y_test = ss_y.transform(Y_test.reshape(-1, 1))

    print('Start SVR ...')
    linear_svr = SVR(kernel='linear')  # 线性核函数初始化的SVR
    linear_svr.fit(X_train, Y_train)
    linear_svr_Y_predict = linear_svr.predict(X_test)

    rbf_svr = SVR(kernel='rbf')  # 径向基核函数初始化的SVR
    rbf_svr.fit(X_train, Y_train)
    rbf_svr_Y_predict = rbf_svr.predict(X_test)

    print('R-squared value of linear SVR is', linear_svr.score(X_test, Y_test))
    print('The mean squared error of linear SVR is', mean_squared_error(ss_y.inverse_transform(Y_test),
                                                                        ss_y.inverse_transform(linear_svr_Y_predict)))
    print('The mean absolute error of linear SVR is', mean_absolute_error(ss_y.inverse_transform(Y_test),
                                                                          ss_y.inverse_transform(linear_svr_Y_predict)))

    print(' ')
    print('R-squared value of RBF SVR is', rbf_svr.score(X_test, Y_test))
    print('The mean squared error of RBF SVR is', mean_squared_error(ss_y.inverse_transform(Y_test),
                                                                     ss_y.inverse_transform(rbf_svr_Y_predict)))
    print('The mean absolute error of RBF SVR is', mean_absolute_error(ss_y.inverse_transform(Y_test),
                                                                       ss_y.inverse_transform(rbf_svr_Y_predict)))
    Y_test = pd.DataFrame(ss_y.inverse_transform(Y_test))
    Y1 = pd.DataFrame(ss_y.inverse_transform(linear_svr_Y_predict))
    Y2 = pd.DataFrame(ss_y.inverse_transform(rbf_svr_Y_predict))
    x = test_df.train_num
    #x = test_df.leav_time
    plt.plot(x, Y_test, c='b', label='target')
    plt.plot(x, Y1, c='r', label='linear')
    plt.plot(x, Y2, c='c', label='rbf')
    plt.show()
    return Y_test


train_dt = ['20141201', '20141202', '20141203', '20141204', '20141205', '20141206', '20141207'
    , '20141208', '20141209', '20141210', '20141211', '20141212', '20141213', '20141214'
    , '20141215', '20141216', '20141217', '20141218', '20141219', '20141220', '20141221'
    , '20141223', '20141224', '20141225', '20141226', '20141227', '20141228']
'''train_dt = ['20141201', '20141202', '20141203', '20141204', '20141205'
    , '20141208', '20141209', '20141210', '20141211', '20141212'
    , '20141215', '20141216', '20141217', '20141218', '20141219'
    , '20141223', '20141224', '20141225', '20141226','20141229']'''
test_dt = '20141229'
mianProcess(train_dt, test_dt)