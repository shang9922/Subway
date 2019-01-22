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


if __name__ == '__main__':
    save_FB_toFile()