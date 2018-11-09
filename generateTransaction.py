#coding:utf-8
import datetime
import pandas as pd

last_card_id = 0
last_time = datetime.datetime.strptime('20141201' + '000000', '%Y%m%d%H%M%S')
last_line_id = 0
last_station_id = 0
last_sign = 0

this_card_id = 0
this_time = datetime.datetime.strptime('20141201' + '000000', '%Y%m%d%H%M%S')
this_line_id = 0
this_station_id = 0
this_sign = 0

ids = []
line_id = []
in_time = []
in_station = []
out_time = []
out_station = []
total_time = []

last_exist = False
i = 0

print 'Start reading data...'
df = pd.read_csv("D:\SubwayData\sj201412_01_07_15972455.csv",usecols=[1,2,3,4,5])
print 'Data has been read yet.'
length = df.shape[0]

print 'Start to generate transactions...'
for x in range(length):
    process = float(x * 100) / length
    print 'process : %.2f %%' % process
    this_sign = df.iloc[x][4]                       # 获取当前记录的进出标志
    if this_sign == 21:
        last_exist = True
        last_card_id = df.iloc[x][0]                # 缓存当前记录
        last_time = df.iloc[x][1]
        last_line_id = df.iloc[x][2]
        last_station_id = df.iloc[x][3]
        # last_sign = this_sign
    else:                                          # this_sign = 22
        this_card_id = df.iloc[x][0]
        this_time = df.iloc[x][1]
        this_line_id = df.iloc[x][2]
        this_station_id = df.iloc[x][3]
        if last_exist and last_line_id == 1 and this_line_id == 1 and last_station_id != this_station_id and last_card_id == this_card_id:
            ids.append(last_card_id)
            line_id.append(last_line_id)
            in_time.append(last_time)
            in_station.append(last_station_id)
            out_time.append(this_time)
            out_station.append(this_station_id)
            time_1 = datetime.datetime.strptime(last_time, '%Y-%m-%d %H:%M:%S')
            time_2 = datetime.datetime.strptime(this_time, '%Y-%m-%d %H:%M:%S')
            total_time.append((time_2 - time_1).seconds)
            i += 1
        last_exist = False

print 'Finish generate transactions, end with %d transactions in total.' % i

write_file = pd.DataFrame({'id': ids,
                           'line_id': line_id,
                           'in_station': in_station,
                           'out_station': out_station,
                           'in_time': in_time,
                           'out_time': out_time,
                           'total_time': total_time}, columns=['id', 'line_id', 'in_station', 'out_station', 'in_time', 'out_time', 'total_time'])
print write_file.head(20)
print 'Start to write data...'
write_file.to_csv('D:\SubwayData\Transactions_201412_01_07_line_1_1276913.csv')