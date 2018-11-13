#coding:utf-8
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN

print 'Start reading data...'
df = pd.read_csv("D:\SubwayData\Transactions_201412_01_07_line_1_1276913.csv", usecols=[3, 4, 5, 7])
print 'Data has been read yet.'

star_time = datetime.datetime.strptime('20141205' + '160000', '%Y%m%d%H%M%S')
end_time = datetime.datetime.strptime('20141205' + '173000', '%Y%m%d%H%M%S')

df = df[(pd.to_datetime(df.in_time) >= star_time) & (pd.to_datetime(df.in_time) <= end_time) & (df.in_station == 23) & (
df.out_station == 14) & (df.total_time < 3000)& (df.total_time > 1300)].loc[:, ['in_time', 'total_time']]

x = []
y = df['total_time']
for i in df['in_time']:
    x.append((datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S')-star_time).seconds)
df = pd.DataFrame({ 'in_time' : x,
                    'total_time' : y})
X = df.values

y_pred = DBSCAN(eps = 90, min_samples = 7).fit_predict(X)
s = pd.Series(y_pred)
X = pd.DataFrame(X)
X['C'] = s
#plt.scatter(X[X.C!=-1].loc[:,0], X[X.C!=-1].loc[:,1], c=X[X.C!=-1].loc[:,'C'])
#plt.show()

n_clusters_ = len(set(y_pred))
mins = []

for i in range(n_clusters_-1):#排除分类标签-1的数据，所以实际分类数量要-1
    tra_time = X[X.C==i].loc[:,1]
    mins.append(tra_time.min(0))

print min(mins)
min_total = min(mins)
df['new_x'] = df.in_time + (df.total_time - min_total)
df['new_y'] = 0.75 * min_total + 0.25 * df.total_time
#plt.scatter(df['new_x'], df['new_y'], color='k', s=10, marker='.')
#plt.show()
X = df.loc[:,['new_x','new_y']].values
y_pred = DBSCAN(eps = 90, min_samples = 12).fit_predict(X)
s = pd.Series(y_pred)
df =pd.DataFrame(df.loc[:,['in_time','total_time']].values)#这一步重置DF不可省略，重置DF索引，使之为0,1,2,3.。。与s保持一致，才可以合并
df['C'] = s
plt.scatter(df[df.C!=-1].loc[:,0], df[df.C!=-1].loc[:,1], c=df[df.C!=-1].loc[:,'C'])
plt.show()