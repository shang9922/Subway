#coding:utf-8
import pandas as pd
import numpy as np
import datetime

star_time = datetime.datetime.strptime('20141207' + '063000', '%Y%m%d%H%M%S')
end_time = datetime.datetime.strptime('20141207' + '235900', '%Y%m%d%H%M%S')

df_origin = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_01_07_line_1_1276913.csv", usecols=[3, 4, 5, 7])
count_origin = df_origin[(df_origin.in_station == 14) & (df_origin.out_station < 14) & (pd.to_datetime(df_origin.in_time) >= star_time) & (pd.to_datetime(df_origin.in_time) <= end_time)].shape[0]
df_clust = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\clusteResult_for14_line1_20141201-07.csv")
count_clust = df_clust[(df_clust.in_id == 14) & (df_clust.out_id < 14) & (pd.to_datetime(df_clust.in_time) >= star_time) & (pd.to_datetime(df_clust.in_time) <= end_time)].shape[0]

print float(count_clust)/count_origin