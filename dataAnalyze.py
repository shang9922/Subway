#coding:utf-8
import datetime
import pandas as pd
import matplotlib.pyplot as plt

print 'Start reading data...'
df = pd.read_csv("D:\SubwayData\Transactions_201412_01_07_line_1_1276913.csv",usecols=[3,4,5,7])
print 'Data has been read yet.'

star_time = datetime.datetime.strptime('20141205' + '160000', '%Y%m%d%H%M%S')
end_time = datetime.datetime.strptime('20141205' + '173000', '%Y%m%d%H%M%S')

df = df[(pd.to_datetime(df.in_time)>=star_time)&(pd.to_datetime(df.in_time)<=end_time)&(df.in_station==23)&(df.out_station==14)&(df.total_time<3500)&(df.total_time>1300)].loc[:,['in_time','total_time']]

#datetime.datetime.strptime('2014-12-01 08:30:24', '%Y-%m-%d %H:%M:%S').time()
x = []
y = df['total_time']
for i in df['in_time']:
    x.append(datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S').time())

plt.scatter(x, y, color='k', s=10, marker='.')
plt.xlabel('Checkin time')
plt.ylabel('Travel time(s)')
plt.title('Travel time and checkin time on 2015-05-08')
plt.xticks( ['16:00:00', '16:15:00', '16:30:00', '16:45:00', '17:00:00', '17:15:00', '17:30:00'])
plt.show()