#coding:utf-8
import csv
import string
import datetime

'''
    对原始数据做基本处理
    由txt格式转为CSV格式
    提取出1号线的数据
    删除不需要的属性
    过滤存在异常值的数据
'''

readFile = open("D:\SubwayData\sj201412.txt",'r')
writeFile = open("D:\SubwayData\sj201412.csv","wb")            #打开方式为wb，否则会出现间隔空行的情况

writer = csv.writer(writeFile)
writer.writerow(['cardID', 'dateAndTime', 'lineID','stationID','inOrOut'])     #表头
i = 0                                             #计数标志

for line in readFile:
    #line = f.readline()
    #line = '"1111111111111111"\t"20140103"\t"23642"\t"214"\t"88"\t"22"\t"\n"'      #原始数据格式
    line = line.replace('"','')                   #删除自带的"号
    line = line.split('\t')                       #分割字符串
    stationID = string.atoi(line[3])              #站点ID转换为整数
    lineID = stationID / 100                      #获取站点所属线路
    line[2] = '%06d' %(string.atoi(line[2]))     #时间靠右对齐，前方补0，以便转换为时间戳
    dateAndTime = datetime.datetime.strptime(line[1] + line[2],'%Y%m%d%H%M%S')     #字符串转为时间戳
    openTime = datetime.datetime.strptime(line[1] + '060000','%Y%m%d%H%M%S')      #地铁开始运营的时间

    if lineID == 1 and dateAndTime >= openTime:      #线路为1号线且刷卡时间无异常
        line.pop()                                   #删除末尾的换乘标志属性
        line.pop(4)                                  #删除卡类型属性
        i = i + 1
        print line[1], i                            #显示进度
        line[0] = string.atoi(line[0])               #卡ID转换为整数
        line[1] = line[1] + ' ' + line[2]            #时间戳
        line[2] = lineID                             #线路号
        line[3] = stationID % 100                    #除去线路号的站台ID
        line[4] = string.atoi(line[4])               #进出站标志转换为整数

        writer.writerow(line)                        #整理后的数据写入CSV文件

writeFile.close()
readFile.close()