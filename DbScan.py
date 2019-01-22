#coding:utf-8
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import DBSCAN
'''
    利用两次DBSACN对事务进行聚类
    第一次初步剔除噪声，并获得最短事务耗时
    第二次变换坐标后得到真正的聚类结果
'''

print 'Start reading data...'
df = pd.read_csv("D:\SubwayData\Transactions_201412_01_07_line_1_1276913.csv", usecols=[3, 4, 5, 7])
print 'Data has been read yet.'

s_time = datetime.datetime.strptime('20141205' + '160000', '%Y%m%d%H%M%S')
e_time = datetime.datetime.strptime('20141205' + '173000', '%Y%m%d%H%M%S')

star_time = datetime.datetime.strptime('20141205' + '063000', '%Y%m%d%H%M%S')
end_time = datetime.datetime.strptime('20141205' + '235900', '%Y%m%d%H%M%S')

df = df[(pd.to_datetime(df.in_time) >= star_time) & (pd.to_datetime(df.in_time) <= end_time) & (df.in_station == 23) & (
df.out_station == 14)] .loc[:, ['in_time', 'total_time']]

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
# plt.scatter(X[X.C!=-1].loc[:,0], X[X.C!=-1].loc[:,1], c=X[X.C!=-1].loc[:,'C'])
# plt.show()

n_clusters_ = len(set(y_pred))
mins = []

for i in range(n_clusters_-1):#排除分类标签-1的数据，所以实际分类数量要-1
    tra_time = X[X.C==i].loc[:,1]
    #mins.append(tra_time.min(0))
    mins.append(min(tra_time))

min_total = min(mins)
df['new_x'] = df.in_time + (df.total_time - min_total)
df['new_y'] = 0.75 * min_total + 0.25 * df.total_time
X = df.loc[:,['new_x','new_y']].values
y_pred = DBSCAN(eps = 90, min_samples = 12).fit_predict(X)
s = pd.Series(y_pred)
df =pd.DataFrame(df.loc[:,['in_time','total_time']].values)#这一步重置DF不可省略，重置DF索引，使之为0,1,2,3.。。与s保持一致，才可以合并
df['C'] = s
df.columns = ['in_time','total_time','C']
df = df[(df.in_time>=(s_time-star_time).seconds)&(df.in_time<=(e_time-star_time).seconds)]
x = []
for i in df['in_time']:
    x.append((datetime.datetime.strptime('20141205' + '063000', '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(i))).time())
df['in_time'] = x
x = df[df.C!=-1].loc[:,'in_time'].values
plt.scatter(x, df[df.C!=-1].loc[:,'total_time'], c=df[df.C!=-1].loc[:,'C'], s=10, marker='.')
plt.xlabel('Checkin time')
plt.ylabel('Travel time(s)')
plt.title('Travel time and checkin time on 2015-05-08')
plt.xticks(['16:00:00', '16:15:00', '16:30:00', '16:45:00', '17:00:00', '17:15:00', '17:30:00'])
plt.yticks([1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500])
plt.show()
# plt.scatter(df.loc[:,0], df.loc[:,1], c=df.loc[:,'C'])
# plt.show()