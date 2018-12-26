# coding:utf-8
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import ShortTermPredict as STP
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

test_dt = '20141229'
dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
    , '20141208', '20141209', '20141210', '20141211', '20141212'
    , '20141215', '20141216', '20141217', '20141218', '20141219'
    , '20141223', '20141225', '20141226']
predict_time_table = STP.predict_by_constant_weights(dts, test_dt, 3)
div = 200000
rs = '0'
for i in dts:
    real_train_data = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_' + i + '.csv')
    real_time_table = real_train_data.loc[4:, 'leav_time'].copy()  # 真实的时刻表
    real_time_table *= 15
    mean_squared_error(test['count'], ha)
