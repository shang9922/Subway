#coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
'''
    初步得到时刻表，和粗略的leav_seconds
'''
date_str = '20141205'
star_time = datetime.datetime.strptime(date_str + '063000', '%Y%m%d%H%M%S')
end_time = datetime.datetime.strptime(date_str + '233000', '%Y%m%d%H%M%S')
walk_time = 51.23

print 'Start reading data...'
df = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\Transactions_201412_01_07_line_1_1276913.csv", usecols=[3, 4, 5, 7])

st = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\shortTravelTime\shortTravelTime_for14_line1_' + date_str + '.csv')
ttr = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\TimeTable\TimeTable_for14_line1_' + date_str +'.csv', usecols=[1])
print 'Data has been read yet.'

df = df[(pd.to_datetime(df.in_time) >= star_time) & (pd.to_datetime(df.in_time) <= end_time) & (df.in_station == 14) & (df.out_station < 14)]
in_seconds = []
for i in df['in_time']:
    in_seconds.append((datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S') - star_time).seconds)
df = pd.DataFrame(df.values, columns=["in_station", "out_station", "in_time", "total_time"])
df['in_seconds'] = in_seconds
df['arr_seconds'] = df['in_seconds'] + walk_time
count_before = df.shape[0]

df2 = pd.DataFrame(columns=["in_station", "out_station", "in_time", "total_time", "in_seconds", "arr_seconds", "leav_seconds", "wait_seconds"])

for i in range(12):     #过滤掉总耗时小于平均最短耗时的事务，同时计算每个事务对应的离开的秒数，以及等待的秒数
    out_id = 13 - i
    st_time = st.iloc[0][i+3]
    if st_time > 0:
        temp_df = df[(df.out_station == out_id) & (df.total_time >= st_time)]
        temp_df['leav_seconds'] = temp_df['arr_seconds'] + temp_df['total_time'] - st_time
        temp_df['wait_seconds'] = temp_df['total_time'] - st_time
        temp_df.columns = ["in_station", "out_station", "in_time",  "total_time", "in_seconds", "arr_seconds", "leav_seconds", "wait_seconds"]
        df2 = df2.append(temp_df, ignore_index=True)

for j in range(ttr.shape[0]): #个别时刻没对准到15s整数倍，将其对整
    temp = ttr.iloc[j, 0]
    yu = temp % 15
    if yu != 0:
        ttr.iloc[j, 0] = temp - yu

'''根据时刻表，重新校正leav_seconds，wait_seconds'''
for i in range(df2.shape[0]):
    arr_seconds = df2.iloc[i, 5]
    leav_seconds = df2.iloc[i, 6]
    wait_seconds = df2.iloc[i, 7]
    for j in range(ttr.shape[0]):
        if j == 0 and leav_seconds <= ttr.iloc[0, 0]:
            leav_seconds = ttr.iloc[0, 0] + 3
            break
        elif j == ttr.shape[0] - 1 and leav_seconds >= ttr.iloc[j, 0]:
            leav_seconds = ttr.iloc[j, 0] + 3
            break
        elif ttr.iloc[j, 0] <= leav_seconds < ttr.iloc[j+1, 0]:
            before = leav_seconds - ttr.iloc[j, 0]
            after = ttr.iloc[j+1, 0] - leav_seconds
            if before < after:  #距离前一个检查点更近
                leav_seconds = ttr.iloc[j, 0] + 3
                if leav_seconds < arr_seconds:
                    leav_seconds = ttr.iloc[j + 1, 0] + 3 #wait_seconds不能为负
            else:   #距离后一个检查点更近
                leav_seconds = ttr.iloc[j+1, 0] + 3
            break
    wait_seconds = leav_seconds - arr_seconds
    df2.iloc[i, 6] = leav_seconds
    df2.iloc[i, 7] = wait_seconds
df2.to_csv('E:\Pycharm\PythonProjects\Subway\data\WaitTime\waitTime_for14_line1_' + date_str + '.csv')



'''
    绘图，横轴arr_seconds，纵轴wait_seconds
    只筛选了用于作图的点，实际数据未做筛选
'''
'''
temp = df2[(df2.in_seconds > 0) & (df2.in_seconds < 9000) & (df2.wait_seconds < 600)]
x = temp['arr_seconds']
#x = []
#for i in temp['in_time']:
#    x.append(datetime.datetime.strptime(i, '%Y-%m-%d %H:%M:%S').time())
plt.scatter(x, temp['wait_seconds'], color='k', s=10, marker='.')
plt.show()'''

'''统计实时站台人数'''
'''
x = []
y = []
for i in range(1, 601):
    check_seconds = i * 15
    x.append(check_seconds)
    count = df2[(df2.arr_seconds <= check_seconds) & (df2.leav_seconds > check_seconds)].shape[0]
    #count = df2[(df2.in_seconds > (check_seconds - 180)) & (df2.in_seconds <= check_seconds)].shape[0]
    y.append(count)'''
#plt.plot(x, y)
#plt.show()

'''在实时人数序列中找极大值点'''
'''
t = 5
u = 5
x2 = []
y2 = []

for i in range(t, len(x)-u):
    before = []
    after = []
    for p in range(1, t+1):
        before.append(y[i-p])
    for q in range(1, u+1):
        after.append(y[i+q])
    if y[i] >= max(before) and y[i] > max(after):
        x2.append(x[i])
        y2.append(y[i])

rs = pd.DataFrame({'check_seconds': x2}, columns=['check_seconds'])
#rs.to_csv('E:\Pycharm\PythonProjects\Subway\data\TimeTable\TimeTable_temp_for14_line1_' + date_str + '.csv')

plt.scatter(x2, y2, color='k', marker='o')
plt.plot(x, y)
plt.show()'''
#df3 = pd.DataFrame({'check_seconds': x,
#                    'count_in_platform': y
#                       }, columns=['check_seconds', 'count_in_platform'])
#df3.to_csv('E:\Pycharm\PythonProjects\Subway\data\platformCount_for14_line1_20141201.csv')

