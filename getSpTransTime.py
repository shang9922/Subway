#coding:utf-8
import pandas as pd
import numpy as np
import datetime
from sklearn.cluster import DBSCAN


def getRusultDf(dataf, station_id, dateandtime):
    hang = 23 - station_id + 1
    lie  = station_id
    X = np.zeros([hang, lie], dtype = np.int)
    cols = ['in_station']
    total = hang * (lie - 1)
    p = 0
    for h in range(hang):
        in_id = station_id + h
        X[h, 0] = in_id
        for l in range(lie-1):
            out_id = station_id - l
            if (in_id) != (out_id):
                X[h, l+1] = getAvgTime(in_id, out_id, dateandtime, dataf)
            p+=1
            process = float(p * 100) / total
            print 'process : %.2f %%' % process
    for c in range(lie-1):
        out_id = station_id - c
        cols.append(out_id)
    X = pd.DataFrame(X)
    X.columns = cols
    return X

def getAvgTime(in_id, out_id, dateandtime, dataf):
    times = []
    for x in dateandtime:
        temp = dfSelect(in_id, out_id, x, dataf)
        if temp != -1:
           times.append(temp)
    if (in_id == 15) & (out_id == 10):
        print times
    if times:
        return np.mean(times)
    else:
        return -1

def dfSelect(in_id, out_id, dateandtime, dataf):
    star_time = datetime.datetime.strptime(dateandtime + '063000', '%Y%m%d%H%M%S')
    end_time = datetime.datetime.strptime(dateandtime + '235900', '%Y%m%d%H%M%S')
    dataf = dataf[(pd.to_datetime(dataf.in_time) >= star_time) & (pd.to_datetime(dataf.in_time) <= end_time) & (dataf.in_station == in_id) & (dataf.out_station == out_id)].loc[:,['in_time', 'total_time']]
    x = []
    y = dataf['total_time']
    for i in dataf['in_time']:
        x.append((datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') - star_time).seconds)
    dataf = pd.DataFrame({'in_time': x,
                       'total_time': y})
    result = getShortTime(dataf)
    return result

def getShortTime(x):
    if x.shape[0] == 0:
        return -1
    X = x.values
    y_pred = DBSCAN(eps=90, min_samples=7).fit_predict(X)
    s = pd.Series(y_pred)
    X = pd.DataFrame(X)
    X['C'] = s
    n_clusters_ = len(set(y_pred))
    mins = []

    if n_clusters_ <= 5:  #小于10个聚类，说明数据不足，放弃该站对应数据
        return  -1

    for i in range(n_clusters_ - 1):  # 排除分类标签-1的数据，所以实际分类数量要-1
        tra_time = X[X.C == i].loc[:, 1]
        mins.append(min(tra_time))
    if mins:
        return np.mean(mins)
    else:
        return -1


print 'Start reading data...'
df = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_01_07_line_1_1276913.csv", usecols=[3, 4, 5, 7])
print 'Data has been read yet.'

#dt = ['20141201', '20141202', '20141203', '20141204', '20141205', '20141206', '20141207']
dt = ['20141205']
station_id = 14
resultDF = getRusultDf(df, station_id, dt)
print resultDF
#resultDF.to_csv('E:\Pycharm\PythonProjects\Subway\shorTravelTime_for14_line1_20141201-07-2.csv')