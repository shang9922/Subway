# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error,mean_squared_error


# 获得指定起班次之间的历史平均间隔（秒）
def get_avg_duration(start, end, dts):
    durations = []
    for date_str in dts:                # 分别计算每天的间隔时间
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + date_str + '.csv').loc[:, ['train_num', 'leav_time']]
        if start < temp_df.shape[0] and end <= temp_df.shape[0]:
            derta = (temp_df.iloc[end-1, 1] - temp_df.iloc[start-1, 1]) * 15    # 还原成秒数
            durations.append(derta)
    return np.mean(durations)


# 获得K步预测在历史数据中的平均方差
def get_fix_K_mean_squared_error(k, dts, df):
    train = []
    test = []
    for date_str in dts:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + date_str + '.csv').loc[:, ['train_num', 'leav_time']]
        if k < temp_df.shape[0]:
            for i in range(temp_df.shape[0] - k):
                start_seconds = temp_df.iloc[i, 1] * 15    # 还原成秒数
                train.append(K_predict(i+1, start_seconds, k, df))
                test.append(temp_df.iloc[i+k, 1] * 15)
        else:
            return -1
    return mean_squared_error(test, train)


# K步预测
def K_predict(start_num, start_seconds, k, df):
    duration = df[start_num-1, k-1]
    rs = reParse(start_seconds + duration)
    return rs


# 将任意秒数对齐到15的倍数，4舍5入
def reParse(x):
    beishu = int(x/15)
    downer = beishu * 15
    upper = downer + 15
    if x - downer >= 7.5:
        return upper            # 5入
    else:
        return downer           # 4舍


# 计算历史平均间隔时间矩阵，避免重复计算
def get_avg_duration_matric():
    avg_du_df = np.zeros((203, 20), dtype='float64')
    for i in range(203):
        for j in range(20):
            if i + j + 2 <= 204:
                avg_du_df[i, j] = get_avg_duration(i + 1, i + j + 2, dts)
            else:
                avg_du_df[i, j] = -1
    return avg_du_df


# 获取1-20步长预测在训练集上的经验方差
def get_mean_squared_error_matrix(dts):
    avg_du_matric = get_avg_duration_matric()
    X = []
    Y = []
    for i in range(1, 21):
        X.append(i)
        y = get_fix_K_mean_squared_error(i, dts, avg_du_matric)
        Y.append(y)
        print "%d: %f" % (i, y)
    rs_df = pd.DataFrame(pd.Series(Y))
    rs_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\mean_squared_error.csv')
    plt.plot(X, Y)
    plt.show()


# 将历史平均间隔时间矩阵输出到CSV文件中
def output_avg_du_matric_to_CSV():
    avg_du_matric = get_avg_duration_matric()
    avg_du_df = pd.DataFrame(avg_du_matric)
    avg_du_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\Avg_Duration_Matric.csv')


# 计算历史平均时刻表在训练集上的预测均方差
def get_HA_mean_squared_error(dts):
    ha_seconds = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\Avg_Time_Table.csv').check_seconds
    temp = []
    for dt in dts:
        test_seconds = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + dt + '.csv').leav_time * 15
        temp.append(mean_squared_error(test_seconds, ha_seconds))
    return np.mean(temp)


# 固定权重下的预测
def predict_by_constant_weights(dts, test_str):
    ha_seconds = list(pd.read_csv( 'E:\Pycharm\PythonProjects\Subway\data\TrainData\Avg_Time_Table.csv').iloc[:, 2])
    test_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + test_str + '.csv')
    test_seconds = list(test_df.leav_time * 15)
    avg_du_matric = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\TrainData\Avg_Duration_Matric.csv').iloc[:, 1:7].values
    fangcha_ha = get_HA_mean_squared_error(dts)
    fangcha_du = list(pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\TrainData\mean_squared_error.csv').iloc[0:6, 1].values)

    weights = []
    predict_factor = []
    train_index = -3
    predic_now = 4
    predic_rs = []
    while predic_now <= 203:
        del predict_factor[:]  # 清空list
        if predic_now - train_index > 6:    # 需要移动预测滑窗
            train_index = predic_now - 4
        predict_factor.append(test_seconds[train_index] + avg_du_matric[train_index][predic_now - train_index - 1])
        predict_factor.append(test_seconds[train_index + 1] + avg_du_matric[train_index + 1][predic_now - train_index - 2])
        predict_factor.append(test_seconds[train_index + 2] + avg_du_matric[train_index + 2][predic_now - train_index - 3])
        predict_factor.append(test_seconds[train_index + 3] + avg_du_matric[train_index + 3][predic_now - train_index - 4])
        predict_factor.append(ha_seconds[predic_now])

        fenmu = 1/fangcha_ha + 1/fangcha_du[predic_now - train_index - 1] + 1/fangcha_du[predic_now - train_index - 2]\
                + 1/fangcha_du[predic_now - train_index - 3] + 1/fangcha_du[predic_now - train_index - 4]
        del weights[:]  # 清空list
        # 更新权值
        weights.append((1/fangcha_du[predic_now - train_index - 1])/fenmu)
        weights.append((1/fangcha_du[predic_now - train_index - 2])/fenmu)
        weights.append((1/fangcha_du[predic_now - train_index - 3])/fenmu)
        weights.append((1/fangcha_du[predic_now - train_index - 4])/fenmu)
        weights.append((1/fangcha_ha)/fenmu)
        rs = np.dot(predict_factor, weights)
        rs = reParse(rs)
        predic_rs.append(rs)
        predic_now = predic_now + 1
    # print mean_squared_error(test_seconds[4:], predic_rs)
    # print math.sqrt(mean_squared_error(test_seconds[4:], predic_rs))
    # print mean_absolute_error(test_seconds[4:], predic_rs)
    #
    # Y1 = []
    # Y2 = []
    # X = []
    # for i in range(200):
    #     X.append(i+4)
    #     Y1.append(abs(test_seconds[i+4]-predic_rs[i]))
    #     # Y2.append(predic_rs[i] - min(test_seconds[i + 4], predic_rs[i]))
    # plt.plot(X, Y1, c='b', label='real')
    # # plt.plot(X, Y2, c='r', label='pre')
    # plt.show()
    return predic_rs

# 动态权重下的预测
def predict_by_dynamic_weights(dts, test_str):
    ha_seconds = list(pd.read_csv( 'E:\Pycharm\PythonProjects\Subway\data\TrainData\Avg_Time_Table.csv').iloc[:, 2])
    test_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + test_str + '.csv')
    test_seconds = list(test_df.leav_time * 15)
    avg_du_matric = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\TrainData\Avg_Duration_Matric.csv').iloc[:, 1:7].values
    fangcha_ha = get_HA_mean_squared_error(dts)
    fangcha_du = list(pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\TrainData\mean_squared_error.csv').iloc[0:6, 1].values)

    weights = []
    predict_factor = []
    train_index = 0
    predic_now = 4
    predic_rs = []
    fangcha_temp = []
    m = np.zeros((5, 3), dtype='float64')

    fangcha_temp.append(fangcha_du[predic_now - train_index - 1])
    fangcha_temp.append(fangcha_du[predic_now - train_index - 2])
    fangcha_temp.append(fangcha_du[predic_now - train_index - 3])
    fangcha_temp.append(fangcha_du[predic_now - train_index - 4])
    fangcha_temp.append(fangcha_ha)

    fenmu = 1 / fangcha_temp[0] + 1 / fangcha_temp[1] + 1 / fangcha_temp[2] + 1 / fangcha_temp[3] + 1 / fangcha_temp[4]
    del weights[:]  # 清空list
    weights.append((1 / fangcha_temp[0]) / fenmu)
    weights.append((1 / fangcha_temp[1]) / fenmu)
    weights.append((1 / fangcha_temp[2]) / fenmu)
    weights.append((1 / fangcha_temp[3]) / fenmu)
    weights.append((1 / fangcha_temp[4]) / fenmu)

    while predic_now <= 203:
        del predict_factor[:]  # 清空list
        if predic_now - train_index > 6:    # 需要移动预测滑窗
            train_index = predic_now - 4
            del fangcha_temp[:]
            fangcha_temp.append(np.mean(m[0]))
            fangcha_temp.append(np.mean(m[1]))
            fangcha_temp.append(np.mean(m[2]))
            fangcha_temp.append(np.mean(m[3]))
            fangcha_temp.append(max(np.mean(m[4]), 100))
            if fangcha_temp[0] == 0 or fangcha_temp[1] == 0 or fangcha_temp[2] == 0 or fangcha_temp[3] == 0 or fangcha_temp[4] == 0:
                print predic_now
            m = np.zeros((5, 3), dtype='float64')
            fenmu = 1 / fangcha_temp[0] + 1 / fangcha_temp[1] + 1 / fangcha_temp[2] + 1 / fangcha_temp[3] + 1 / fangcha_temp[4]
            del weights[:]  # 清空list
            weights.append((1 / fangcha_temp[0]) / fenmu)
            weights.append((1 / fangcha_temp[1]) / fenmu)
            weights.append((1 / fangcha_temp[2]) / fenmu)
            weights.append((1 / fangcha_temp[3]) / fenmu)
            weights.append((1 / fangcha_temp[4]) / fenmu)

        predict_factor.append(test_seconds[train_index] + avg_du_matric[train_index][predic_now - train_index - 1])
        predict_factor.append(
            test_seconds[train_index + 1] + avg_du_matric[train_index + 1][predic_now - train_index - 2])
        predict_factor.append(
            test_seconds[train_index + 2] + avg_du_matric[train_index + 2][predic_now - train_index - 3])
        predict_factor.append(
            test_seconds[train_index + 3] + avg_du_matric[train_index + 3][predic_now - train_index - 4])
        predict_factor.append(ha_seconds[predic_now])

        rs = np.dot(predict_factor, weights)
        rs = reParse(rs)
        m[0][predic_now - train_index - 4] = np.square(predict_factor[0] - rs)
        m[1][predic_now - train_index - 4] = np.square(predict_factor[1] - rs)
        m[2][predic_now - train_index - 4] = np.square(predict_factor[2] - rs)
        m[3][predic_now - train_index - 4] = np.square(predict_factor[3] - rs)
        m[4][predic_now - train_index - 4] = np.square(predict_factor[4] - rs)
        predic_rs.append(rs)
        predic_now += 1
    print math.sqrt(mean_squared_error(test_seconds[4:], predic_rs))
    print mean_absolute_error(test_seconds[4:], predic_rs)


# dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
#     , '20141208', '20141209', '20141210', '20141211', '20141212'
#     , '20141215', '20141216', '20141217', '20141218', '20141219'
#     , '20141223', '20141225', '20141226']
# predict_by_constant_weights(dts, '20141229')