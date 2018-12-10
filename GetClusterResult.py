#coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
'''
    通过聚类方法，找到与目标站点相关的事务，记录分类标记，存入csv，以便计算平均步行时间
'''

def getClusterRusult(dataf, station_id, dateandtime):
    df = pd.DataFrame(columns=["in_id", "out_id", "in_time", "in_seconds", "total_time", "C"])
    hang = 23 - station_id + 1
    lie  = station_id - 1
    total = hang * lie
    p = 0
    for h in range(hang):
        in_id = station_id + h
        for l in range(lie):
            out_id = station_id - l
            if (in_id) != (out_id):
                temp = clusterById(in_id, out_id, dateandtime, dataf)
                if temp.shape[0]!=0:
                    df = df.append(temp, ignore_index=True)
            p+=1
            process = float(p * 100) / total
            print 'process : %.2f %%' % process
    return df

def clusterById(in_id, out_id, dateandtime, dataf):
    rs = pd.DataFrame(columns=["in _id", "out_id", "in_time", "in_seconds", "total_time", "C"])
    cache = pd.DataFrame(columns=["in_time", "in_seconds", "total_time", "C"])
    for x in dateandtime:
        temp = clusterByDay(in_id, out_id, x, dataf)
        if temp.shape[0] != 0:
            cache = cache.append(temp.iloc[:], ignore_index=True)
    if cache.shape[0]==0:
        return rs
    rs = pd.DataFrame({'in_id': in_id,
                      'out_id': out_id,
                       'in_time': cache['in_time'],
                       'in_seconds': cache['in_seconds'],
                       'total_time': cache['total_time'],
                       'C': cache['C']},columns=['in_id','out_id','in_time','in_seconds','total_time','C'])
    return rs

def clusterByDay(in_id, out_id, dateandtime, dataf):
    rs = pd.DataFrame(columns=[ "in_time", "in_seconds", "total_time", "C"])
    star_time = datetime.datetime.strptime(dateandtime + '063000', '%Y%m%d%H%M%S')
    end_time = datetime.datetime.strptime(dateandtime + '235900', '%Y%m%d%H%M%S')
    dataf = dataf[(pd.to_datetime(dataf.in_time) >= star_time) & (pd.to_datetime(dataf.in_time) <= end_time) & (dataf.in_station == in_id) & (dataf.out_station == out_id)].loc[:,['in_time', 'total_time']]
    x = []
    y = dataf['total_time']
    z = dataf['in_time']
    for i in dataf['in_time']:
        x.append((datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') - star_time).seconds)
    dataf = pd.DataFrame({'in_time': z,
                          'in_seconds': x,
                       'total_time': y})
    if dataf.shape[0] != 0:
        X = dataf.loc[:,['in_seconds', 'total_time']].values
        y_pred = DBSCAN(eps=90, min_samples=7).fit_predict(X)
        s = pd.Series(y_pred)
        X = pd.DataFrame(X)
        X['C'] = s
        n_clusters_ = len(set(y_pred))
        mins = []
        if n_clusters_ > 5:
            for i in range(n_clusters_ - 1):  # 排除分类标签-1的数据，所以实际分类数量要-1
                tra_time = X[X.C == i].loc[:, 1]
                mins.append(min(tra_time))
            min_total = min(mins)
            dataf['new_x'] = dataf.in_seconds + (dataf.total_time - min_total)
            dataf['new_y'] = 0.75 * min_total + 0.25 * dataf.total_time
            X = dataf.loc[:, ['new_x', 'new_y']].values
            y_pred = DBSCAN(eps=90, min_samples=12).fit_predict(X)
            s = pd.Series(y_pred)
            dataf = pd.DataFrame(
                dataf.loc[:, ['in_time','in_seconds', 'total_time']].values)  # 这一步重置DF不可省略，重置DF索引，使之为0,1,2,3.。。与s保持一致，才可以合并
            dataf['C'] = s
            dataf = dataf[dataf.C != -1]
            dataf.columns = ['in_time','in_seconds', 'total_time','C']
            #if in_id==15 and out_id==10 and dateandtime == '20141205':
            #    plt.scatter(dataf[dataf.C != -1].loc[:, 'in_seconds'], dataf[dataf.C != -1].loc[:, 'total_time'], c=dataf[dataf.C != -1].loc[:, 'C'])
            #    plt.show()
            rs = dataf
    return rs

print 'Start reading data...'
df = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_22_31_line_1.csv", usecols=[3, 4, 5, 7])
print 'Data has been read yet.'

dt = ['20141222', '20141223', '20141224', '20141225', '20141226', '20141227', '20141228', '20141229', '20141230', '20141231']
#dt = ['20141205']
station_id = 14
resultDF = getClusterRusult(df, station_id, dt)
print resultDF
resultDF.to_csv('E:\Pycharm\PythonProjects\Subway\data\clusteResult\clusteResult_for14_line1_20141222-31.csv')