#coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import math
import FinalPredict as FP
import FixProportion as FixPro
import Avg_Alight as AA
import Fix_Board as FB
import matplotlib.pyplot as plt


test_dt = '20141229'
dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
    , '20141208', '20141209', '20141210', '20141211', '20141212'
    , '20141215', '20141216', '20141217', '20141218', '20141219'
    , '20141223', '20141225', '20141226']
svr_dt = ['20141201', '20141202', '20141203', '20141204', '20141205', '20141206', '20141207'
    , '20141208', '20141209', '20141210', '20141211', '20141212', '20141213', '20141214'
    , '20141215', '20141216', '20141217', '20141218', '20141219', '20141220', '20141221'
    , '20141223', '20141224', '20141225', '20141226', '20141227', '20141228']


def save_HA_toFile():
    HA_rs = FP.HA_predict()
    check_time = []
    for i in HA_rs.check_seconds:
        check_time.append((datetime.datetime.strptime('20141229063000', '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(i))).time())
    HA_rs['check_time'] = check_time
    HA_rs.to_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\HA.csv')


def save_FP_realTT():
    count = FP.real_time_table_predict(dts, svr_dt)
    check_time = []
    check_seconds = [i*15 for i in range(1, 4081)]
    for i in check_seconds:
        check_time.append((datetime.datetime.strptime('20141229063000', '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(i))).time())
    FP_TT_df = pd.DataFrame({'check_seconds': check_seconds,
                    'count': count,
                    'check_time': check_time},
                   columns=['check_seconds', 'count', 'check_time'])
    FP_TT_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FP_TT.csv')


def sava_FP_toFile():
    raw_count = FP.short_term_predict(dts, svr_dt, test_dt, steps=3)
    real_count = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv').loc[:, ['count']]
    count_len = len(raw_count)
    count = []
    for i in range(4080-count_len):
        count.append(real_count['count'][i])
    for i in range(count_len):
        count.append(raw_count[i])
    check_time = []
    check_seconds = [i * 15 for i in range(1, 4081)]
    for i in check_seconds:
        check_time.append(
            (datetime.datetime.strptime('20141229063000', '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(i))).time())
    FP_df = pd.DataFrame({'check_seconds': check_seconds,
                             'count': count,
                             'check_time': check_time},
                            columns=['check_seconds', 'count', 'check_time'])
    FP_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FP.csv')


def save_FixPro_toFile():
    count = FixPro.get_fix_proportion_predict()
    check_time = []
    check_seconds = [i * 15 for i in range(1, 4081)]
    for i in check_seconds:
        check_time.append(
            (datetime.datetime.strptime('20141229063000', '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(i))).time())
    FixPro_df = pd.DataFrame({'check_seconds': check_seconds,
                             'count': count,
                             'check_time': check_time},
                            columns=['check_seconds', 'count', 'check_time'])
    FixPro_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FixPro.csv')


def save_AA_toFile():
    count = AA.get_avg_alight_predict()
    check_time = []
    check_seconds = [i * 15 for i in range(1, 4081)]
    for i in check_seconds:
        check_time.append(
            (datetime.datetime.strptime('20141229063000', '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(i))).time())
    AA_df = pd.DataFrame({'check_seconds': check_seconds,
                             'count': count,
                             'check_time': check_time},
                            columns=['check_seconds', 'count', 'check_time'])
    AA_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\AA.csv')


def save_FB_toFile():
    count = FB.get_fix_board_predict()
    check_time = []
    check_seconds = [i * 15 for i in range(1, 4081)]
    for i in check_seconds:
        check_time.append(
            (datetime.datetime.strptime('20141229063000', '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(i))).time())
    FB_df = pd.DataFrame({'check_seconds': check_seconds,
                             'count': count,
                             'check_time': check_time},
                            columns=['check_seconds', 'count', 'check_time'])
    FB_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FB.csv')


def mean_absolute_percentage_error(test, predict, star=0):
    middle = np.mean(test)
    rs = []
    for i in range(len(test)):
        fenmu = 0
        if test[i+star] <= 8:
            continue
        else:
            fenmu = test[i+star]
        fenzi = abs(test[i+star] - predict[i+star])
        rs.append(float(fenzi)/fenmu)
    return np.mean(rs)


def weighted_mean_absolute_error(test, predict):
    fenmu = max(test)
    rs = []
    for i in range(len(test)):
        if test[i] == 0:
            p = 1
        else:
            p = test[i]
        fenzi = (abs(test[i] - predict[i]))*p*p
        rs.append(float(fenzi)/fenmu)
    return np.mean(rs)


def step_tu():
    HA = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\HA.csv').loc[:,['count']]
    FixPro = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FixPro.csv').loc[:,['count']]
    AA = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\AA.csv').loc[:,['count']]
    FB = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FB.csv').loc[:,['count']]
    FP = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FP.csv').loc[:,['count']]
    count = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv').loc[:,['count']]
    MAE = np.zeros([5, 10])
    RMSE = np.zeros([5, 10])
    R = np.zeros([5, 10])
    MAPE = np.zeros([5, 10])
    WMAE = np.zeros([5, 10])
    X = []
    for i in range(10):
        star = 400*i
        derta = 400
        if i == 9:
            derta = 480
        X.append(star + derta)
        MAE[0, i] = mean_absolute_error(count['count'][0:star + derta], HA['count'][0:star + derta])
        MAE[1, i] = mean_absolute_error(count['count'][0:star + derta], FixPro['count'][0:star + derta])
        MAE[2, i] = mean_absolute_error(count['count'][0:star + derta], AA['count'][0:star + derta])
        MAE[3, i] = mean_absolute_error(count['count'][0:star + derta], FB['count'][0:star + derta])
        MAE[4, i] = mean_absolute_error(count['count'][0:star + derta], FP['count'][0:star + derta])

        RMSE[0, i] = math.sqrt(mean_squared_error(count['count'][0:star + derta], HA['count'][0:star + derta]))
        RMSE[1, i] = math.sqrt(mean_squared_error(count['count'][0:star + derta], FixPro['count'][0:star + derta]))
        RMSE[2, i] = math.sqrt(mean_squared_error(count['count'][0:star + derta], AA['count'][0:star + derta]))
        RMSE[3, i] = math.sqrt(mean_squared_error(count['count'][0:star + derta], FB['count'][0:star + derta]))
        RMSE[4, i] = math.sqrt(mean_squared_error(count['count'][0:star + derta], FP['count'][0:star + derta]))

        R[0, i] = r2_score(count['count'][0:star + derta], HA['count'][0:star + derta])
        R[1, i] = r2_score(count['count'][0:star + derta], FixPro['count'][0:star + derta])
        R[2, i] = r2_score(count['count'][0:star + derta], AA['count'][0:star + derta])
        R[3, i] = r2_score(count['count'][0:star + derta], FB['count'][0:star + derta])
        R[4, i] = r2_score(count['count'][0:star + derta], FP['count'][0:star + derta])

        MAPE[0, i] = mean_absolute_percentage_error(count['count'][0:star + derta], HA['count'][0:star + derta])
        MAPE[1, i] = mean_absolute_percentage_error(count['count'][0:star + derta], FixPro['count'][0:star + derta])
        MAPE[2, i] = mean_absolute_percentage_error(count['count'][0:star + derta], AA['count'][0:star + derta])
        MAPE[3, i] = mean_absolute_percentage_error(count['count'][0:star + derta], FB['count'][0:star + derta])
        MAPE[4, i] = mean_absolute_percentage_error(count['count'][0:star + derta], FP['count'][0:star + derta])

        # WMAE[0, i] = weighted_mean_absolute_error(count['count'][0:star + derta], HA['count'][0:star + derta])
        # WMAE[1, i] = weighted_mean_absolute_error(count['count'][0:star + derta], FixPro['count'][0:star + derta])
        # WMAE[2, i] = weighted_mean_absolute_error(count['count'][0:star + derta], AA['count'][0:star + derta])
        # WMAE[3, i] = weighted_mean_absolute_error(count['count'][0:star + derta], FB['count'][0:star + derta])
        # WMAE[4, i] = weighted_mean_absolute_error(count['count'][0:star + derta], FP['count'][0:star + derta])

    plt.plot(X, MAE[0], c='k', label='HA', marker='o')
    plt.plot(X, MAE[1], c='b', label='MIFBP-TP', marker='v')
    # plt.plot(X, MAE[2], c='m', label='SAN', marker='^')
    plt.plot(X, MAE[3], c='g', label='FBP', marker='x')
    plt.plot(X, MAE[4], c='r', label='MIFBP', marker='s')
    plt.title('Mean Absolute Error and Predict Length')
    plt.xlabel('Number of Time Slices')
    plt.ylabel('MAE(s)')
    plt.xticks([400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4080])
    plt.legend()
    plt.show()

    plt.plot(X, RMSE[0], c='k', label='HA', marker='o')
    plt.plot(X, RMSE[1], c='b', label='MIFBP-TP', marker='v')
    # plt.plot(X, RMSE[2], c='m', label='SAN', marker='^')
    plt.plot(X, RMSE[3], c='g', label='FBP', marker='x')
    plt.plot(X, RMSE[4], c='r', label='MIFBP', marker='s')
    plt.title('Root Mean Squared Error and Predict Length')
    plt.xlabel('Number of Time Slices')
    plt.ylabel('RMSE(s)')
    plt.xticks([400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4080])
    plt.legend()
    plt.show()

    plt.plot(X, R[0], c='k', label='HA', marker='o')
    plt.plot(X, R[1], c='b', label='MIFBP-TP', marker='v')
    # plt.plot(X, R[2], c='m', label='SAN', marker='^')
    plt.plot(X, R[3], c='g', label='FBP', marker='x')
    plt.plot(X, R[4], c='r', label='MIFBP', marker='s')
    plt.title('R-squared and Predict Length')
    plt.xlabel('Number of Time Slices')
    plt.ylabel('R-squared')
    plt.xticks([400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4080])
    plt.legend(loc='lower left', bbox_to_anchor=(0, 0.2))
    plt.show()

    plt.plot(X, MAPE[0], c='k', label='HA', marker='o')
    plt.plot(X, MAPE[1], c='b', label='MIFBP-TP', marker='v')
    # plt.plot(X, MAPE[2], c='m', label='SAN', marker='^')
    plt.plot(X, MAPE[3], c='g', label='FBP', marker='x')
    plt.plot(X, MAPE[4], c='r', label='MIFBP', marker='s')
    plt.title('Mean Absolute Percentage Error and Predict Length')
    plt.xlabel('Number of Time Slices')
    plt.ylabel('MAPE')
    plt.xticks([400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4080])
    plt.legend(bbox_to_anchor=(1, 0.6))
    plt.show()

    # plt.plot(X, WMAE[0], c='k', label='HA', marker='o')
    # plt.plot(X, WMAE[1], c='b', label='FixPro', marker='v')
    # plt.plot(X, WMAE[2], c='m', label='AA', marker='^')
    # plt.plot(X, WMAE[3], c='g', label='FB', marker='x')
    # plt.plot(X, WMAE[4], c='r', label='MIFBP', marker='s')
    # plt.title('Weight Mean Absolute Error and Predict Length')
    # plt.xlabel('Number of Time Slices')
    # plt.ylabel('WMAE')
    # plt.xticks([400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4080])
    # plt.legend()
    # plt.show()


def time_tu():
    HA = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\HA.csv').loc[:, ['count']]
    FixPro = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FixPro.csv').loc[:, ['count']]
    AA = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\AA.csv').loc[:, ['count']]
    FB = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FB.csv').loc[:, ['count']]
    FP = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FP_TT.csv').loc[:, ['count']]
    count = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv').loc[:, ['count']]
    MAE = np.zeros([5, 5])
    RMSE = np.zeros([5, 5])
    R = np.zeros([5, 5])
    MAPE = np.zeros([5, 5])
    star = [0, 239, 839, 2519, 3119]
    end = [239, 839, 2519, 3119, 4080]
    X = []
    for i in range(5):
        X.append(i + 1)
        MAE[0, i] = mean_absolute_error(count['count'][star[i]:end[i]], HA['count'][star[i]:end[i]])
        MAE[1, i] = mean_absolute_error(count['count'][star[i]:end[i]], FixPro['count'][star[i]:end[i]])
        MAE[2, i] = mean_absolute_error(count['count'][star[i]:end[i]], AA['count'][star[i]:end[i]])
        MAE[3, i] = mean_absolute_error(count['count'][star[i]:end[i]], FB['count'][star[i]:end[i]])
        MAE[4, i] = mean_absolute_error(count['count'][star[i]:end[i]], FP['count'][star[i]:end[i]])

        RMSE[0, i] = math.sqrt(mean_squared_error(count['count'][star[i]:end[i]], HA['count'][star[i]:end[i]]))
        RMSE[1, i] = math.sqrt(mean_squared_error(count['count'][star[i]:end[i]], FixPro['count'][star[i]:end[i]]))
        RMSE[2, i] = math.sqrt(mean_squared_error(count['count'][star[i]:end[i]], AA['count'][star[i]:end[i]]))
        RMSE[3, i] = math.sqrt(mean_squared_error(count['count'][star[i]:end[i]], FB['count'][star[i]:end[i]]))
        RMSE[4, i] = math.sqrt(mean_squared_error(count['count'][star[i]:end[i]], FP['count'][star[i]:end[i]]))

        R[0, i] = r2_score(count['count'][star[i]:end[i]], HA['count'][star[i]:end[i]])
        R[1, i] = r2_score(count['count'][star[i]:end[i]], FixPro['count'][star[i]:end[i]])
        R[2, i] = r2_score(count['count'][star[i]:end[i]], AA['count'][star[i]:end[i]])
        R[3, i] = r2_score(count['count'][star[i]:end[i]], FB['count'][star[i]:end[i]])
        R[4, i] = r2_score(count['count'][star[i]:end[i]], FP['count'][star[i]:end[i]])

        MAPE[0, i] = mean_absolute_percentage_error(count['count'][star[i]:end[i]], HA['count'][star[i]:end[i]], star[i])
        MAPE[1, i] = mean_absolute_percentage_error(count['count'][star[i]:end[i]], FixPro['count'][star[i]:end[i]], star[i])
        MAPE[2, i] = mean_absolute_percentage_error(count['count'][star[i]:end[i]], AA['count'][star[i]:end[i]], star[i])
        MAPE[3, i] = mean_absolute_percentage_error(count['count'][star[i]:end[i]], FB['count'][star[i]:end[i]], star[i])
        MAPE[4, i] = mean_absolute_percentage_error(count['count'][star[i]:end[i]], FP['count'][star[i]:end[i]], star[i])

    plt.plot(X, MAE[0], c='k', label='HA', marker='o')
    plt.plot(X, MAE[1], c='b', label='FAP', marker='v')
    plt.plot(X, MAE[2], c='m', label='SAN', marker='^')
    plt.plot(X, MAE[3], c='g', label='FBP', marker='x')
    plt.plot(X, MAE[4], c='r', label='MIFBP', marker='s')
    plt.title('Mean Absolute Error and Predict Length')
    plt.xlabel('Segment')
    plt.ylabel('MAE(s)')
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.show()

    plt.plot(X, RMSE[0], c='k', label='HA', marker='o')
    plt.plot(X, RMSE[1], c='b', label='FAP', marker='v')
    plt.plot(X, RMSE[2], c='m', label='SAN', marker='^')
    plt.plot(X, RMSE[3], c='g', label='FBP', marker='x')
    plt.plot(X, RMSE[4], c='r', label='MIFBP', marker='s')
    plt.title('Root Mean Squared Error and Predict Length')
    plt.xlabel('Segment')
    plt.ylabel('RMSE(s)')
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.show()

    plt.plot(X, R[0], c='k', label='HA', marker='o')
    plt.plot(X, R[1], c='b', label='FAP', marker='v')
    plt.plot(X, R[2], c='m', label='SAN', marker='^')
    plt.plot(X, R[3], c='g', label='FBP', marker='x')
    plt.plot(X, R[4], c='r', label='MIFBP', marker='s')
    plt.title('R-squared and Predict Length')
    plt.xlabel('Segment')
    plt.ylabel('R-squared')
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend(loc='lower right')
    plt.show()

    plt.plot(X, MAPE[0], c='k', label='HA', marker='o')
    plt.plot(X, MAPE[1], c='b', label='FAP', marker='v')
    plt.plot(X, MAPE[2], c='m', label='SAN', marker='^')
    plt.plot(X, MAPE[3], c='g', label='FBP', marker='x')
    plt.plot(X, MAPE[4], c='r', label='MIFBP', marker='s')
    plt.title('Mean Absolute Percentage Error and Predict Length')
    plt.xlabel('Segment')
    plt.ylabel('MAPE')
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend()
    plt.show()


def predict_and_real():
    predict_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PredictResult\FP.csv')
    real_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_20141229.csv')
    check_time = []
    for i in real_df['check_seconds']:
        check_time.append(
            (datetime.datetime.strptime('20141229063000', '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(i))).time())
    real_df['check_time'] = check_time
    predict_df['check_time'] = check_time
    predict_df = predict_df[(predict_df.check_time >= datetime.datetime.strptime('20141229074000', '%Y%m%d%H%M%S').time())
                            & (predict_df.check_time <= datetime.datetime.strptime('20141229083500', '%Y%m%d%H%M%S').time())]
    real_df = real_df[(real_df.check_time >= datetime.datetime.strptime('20141229074000', '%Y%m%d%H%M%S').time())
                            & (real_df.check_time <= datetime.datetime.strptime('20141229083500', '%Y%m%d%H%M%S').time())]
    plt.plot(real_df['check_time'], real_df['count'], c='b', label='Real Data')
    plt.plot(predict_df['check_time'], predict_df['count'], c='r', label='MIFBP-TP')
    plt.title('Assembling Passengers and Time')
    plt.xlabel('Time')
    plt.ylabel('Number of Assembling Passengers')
    plt.xticks([ '07:45:00', '08:00:00', '08:15:00', '08:30:00'])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    step_tu()