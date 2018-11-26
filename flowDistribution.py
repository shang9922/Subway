#coding:utf-8
import pandas as pd
import numpy as np
import datetime
'''
    分析1号线在某段时间内，某个方向上，各站点间的流量比例分布情况
'''

print 'Start reading data...'
df = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_01_07_line_1_1276913.csv", usecols=[3, 4, 5])
print 'Data has been read yet.'

star_time = datetime.datetime.strptime('20141201' + '063000', '%Y%m%d%H%M%S')
end_time = datetime.datetime.strptime('20141207' + '235900', '%Y%m%d%H%M%S')

length = df[(df.in_station > df.out_station) & (pd.to_datetime(df.in_time) >= star_time) & (pd.to_datetime(df.in_time) <= end_time)].shape[0]
print length

X = np.zeros([21, 23], dtype = np.float)

for i in range(21):
    total = 0.0
    in_id = 23-i
    for j in range(i, 21):
        out_id = 22-j
        c = df[(df.in_station == in_id) & (df.out_station == out_id) & (pd.to_datetime(df.in_time) >= star_time) & (pd.to_datetime(df.in_time) <= end_time)].shape[0]
        distrib = float(c * 100) / length
        X[i, j] = distrib
        total = total + distrib
    X[i, 21] = total
    X[i, 22] = in_id

df = pd.DataFrame(X)
colm = []
for i in range(21):
    colm.append(22-i)
colm.append('sum')
colm.append('in_station')
df.columns = colm

df.to_csv('E:\Pycharm\PythonProjects\Subway\data\ditribution_down_line1_20141201-07.csv')