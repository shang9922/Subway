# coding:utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

def mainProcess(dts, total_train = 204):
    train_df = pd.DataFrame(columns=['train_num', 'check_seconds'])

    print('Combining data ...')
    for date_str in dts:                    # 拼接不同天的时刻表数据
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + date_str + '.csv').loc[:, ['train_num', 'leav_time']]
        for i in range(temp_df.shape[0]):
            temp_df.iloc[i, 1] *= 15  # 还原成秒数
        temp_df.columns = ['train_num', 'check_seconds']
        train_df = train_df.append(temp_df, ignore_index=True)

    num = []
    avg_seconds = []
    for i in range(1, total_train + 1):
        num.append(i)
        temp = train_df[train_df.train_num == i]
        prima_avg = np.mean(temp['check_seconds'])      # 直接计算出的平均时刻
        downer = int(prima_avg)/15 * 15
        upper = downer + 15
        zhongShu = np.median(temp['check_seconds'].values)  # 中位数
        if zhongShu >= prima_avg:
            avg_seconds.append(upper)
        else:
            avg_seconds.append(downer)

    rs_df = pd.DataFrame({'train_num': num,
                            'check_seconds': avg_seconds},
                           columns=['train_num', 'check_seconds'])
    rs_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\Avg_Time_Table.csv')

    test_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_20141229.csv').loc[:, ['train_num', 'leav_time']]
    for i in range(test_df.shape[0]):
        test_df.iloc[i, 1] *= 15  # 还原成秒数
    test_df.columns = ['train_num', 'check_seconds']
    print mean_squared_error(test_df.check_seconds, rs_df.check_seconds)
    print mean_absolute_error(test_df.check_seconds, rs_df.check_seconds)
    # plt.plot(rs_df.loc[0:9].train_num, rs_df.loc[0:9].check_seconds, c='r', label='avg')
    # plt.plot(rs_df.loc[0:9].train_num, test_df.loc[0:9].check_seconds, c='b', label='real')
    # plt.show()


dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
    , '20141208', '20141209', '20141210', '20141211', '20141212'
    , '20141215', '20141216', '20141217', '20141218', '20141219'
    , '20141223', '20141224', '20141225', '20141226']
mainProcess(dts)