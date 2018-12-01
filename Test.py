#coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

star_time = datetime.datetime.strptime('20141201' + '073000', '%Y%m%d%H%M%S')
end_time = datetime.datetime.strptime('20141201' + '090000', '%Y%m%d%H%M%S')
walk_time = 29

print 'Start reading data...'
df = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_01_07_line_1_1276913.csv",usecols=[3,4,5,6,7])
print 'Data has been read yet.'

df = df[(pd.to_datetime(df.in_time) >= star_time) & (pd.to_datetime(df.in_time) <= end_time) & (df.in_station ==23) & (df.out_station==14 )]
x = []
y = []
for i in df['in_time']:
    x.append((datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') - star_time).seconds)
for i in df['out_time']:
    x.append((datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') - star_time).seconds)
x = df.out_time
y = df.in_time

plt.scatter(x, y, marker='.')
plt.show()