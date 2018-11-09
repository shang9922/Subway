#coding:utf-8
import string
import datetime
import pandas as pd

'''
    对原始数据做基本处理
    由txt格式转为CSV格式
    删除不需要的属性
    过滤存在异常值的数据
    截取指定时间段的数据
    按卡ID排序，再按时间排序
    为下一步生成乘车事务准备
'''

readFile = open("D:\SubwayData\sj201412.txt", 'r')
writeFile = open("D:\SubwayData\sj201412_without_01_07.txt","w")            #打开方式为wb，否则会出现间隔空行的情

i = 0                                             # 计数标志
ids = []
times = []
lineIds = []
stations = []
inOrOut = []

startTime = datetime.datetime.strptime('20141201' + '060000', '%Y%m%d%H%M%S')    # 截取开始时间
endTime = datetime.datetime.strptime('20141207' + '235959', '%Y%m%d%H%M%S')      # 截取结束时间

for line in readFile:
    #line = '"1111111111111111"\t"20140103"\t"23642"\t"214"\t"88"\t"22"\t"\n"'      # 原始数据格式
    originalLine = line
    line = line.replace('"','')                   # 删除自带的"号
    line = line.split('\t')                       # 分割字符串
    line[2] = '%06d' %(string.atoi(line[2]))     # 时间靠右对齐，前方补0，以便转换为时间戳
    dateAndTime = datetime.datetime.strptime(line[1] + line[2], '%Y%m%d%H%M%S')     # 字符串转为时间戳
    openTime = datetime.datetime.strptime(line[1] + '060000', '%Y%m%d%H%M%S')      # 地铁开始运营的时间

    if dateAndTime >= openTime:
        if dateAndTime >= startTime and dateAndTime <= endTime:      # 刷卡时间无异常
            stationID = string.atoi(line[3])  # 站点ID转换为整数
            lineID = stationID / 100  # 获取站点所属线路
            line.pop()                                   # 删除末尾的换乘标志属性
            line.pop(4)                                  # 删除卡类型属性
            i = i + 1
            print line[1], i                            # 显示进度
            ids.append(string.atoi(line[0]))            # 将该记录按属性存入DataFrame中
            times.append(dateAndTime)
            lineIds.append(lineID)
            stations.append(stationID % 100)
            inOrOut.append(string.atoi(line[4]))
        else:
            print originalLine
            writeFile.write(originalLine)

writeFile.close()
readFile.close()

df = pd.DataFrame({'id':ids,
                   'time':times,
                   'lineId':lineIds,
                   'station':stations,
                   'inOrOut':inOrOut}, columns = ['id','time','lineId','station','inOrOut'])                             # 利用Pandas进行排序等处理
df.sort_values(['id','time'], ascending = True, inplace = True)
print df.head(100)
df.to_csv('D:\SubwayData\sj201412_01_07.csv')
#print df
