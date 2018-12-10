#coding:utf-8
import pandas as pd
import numpy as np
import datetime
'''
    找到与某站点相关的最短事务，并计算平均步行时间
    每天的步行时间最好分开算
'''
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
    #if (in_id == 15) & (out_id == 10):
    #    print times
    if times:
        return np.mean(times)
    else:
        return -1

def dfSelect(in_id, out_id, dateandtime, dataf):
    star_time = datetime.datetime.strptime(dateandtime + '063000', '%Y%m%d%H%M%S')
    end_time = datetime.datetime.strptime(dateandtime + '235900', '%Y%m%d%H%M%S')
    dataf = dataf[(pd.to_datetime(dataf.in_time) >= star_time) & (pd.to_datetime(dataf.in_time) <= end_time) & (dataf.in_id == in_id) & (dataf.out_id == out_id)].loc[:,['in_seconds', 'total_time', 'C']]
    if dataf.shape[0] == 0:
        return -1
    n_clusters_ = len(set(dataf.C))
    mins = []

    for i in range(n_clusters_):
        tra_time = dataf[dataf.C == i].loc[:, 'total_time']
        mins.append(min(tra_time))
    if mins:
        return np.mean(mins)
    else:
        return -1

station_id = 14
dwell = 25.0
print 'Start reading data...'
df = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\clusteResult\clusteResult_for"+ str(station_id) +"_line1_20141222-31.csv")
print 'Data has been read yet.'
#dt = ['20141222', '20141223', '20141224', '20141225', '20141226']
#dt = ['20141201', '20141202', '20141203', '20141204', '20141205']
#dt = ['20141206', '20141207']
#dt = ['20141201', '20141202', '20141203', '20141204']
dt = ['20141229', '20141230', '20141231']
resultDF = getRusultDf(df, station_id, dt)
print resultDF
#resultDF.to_csv('E:\Pycharm\PythonProjects\Subway\data\shortTravelTime\shortTravelTime_for14_line1_20141231.csv')
temps = []
for i in range(1, station_id):
    x = resultDF.iloc[0, i]
    if x > 0:
        for j in range(23 - station_id + 1):
            y = resultDF.iloc[j, 1]
            z = resultDF.iloc[j, i]
            if y > 0 and z > 0:
                temp = (x + y - z + dwell)/2
                temps.append(temp)
if len(temps) > 0:
    print np.mean(temps)
else:
    print "None"