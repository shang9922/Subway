#coding:utf-8
import pandas as pd

papa=pd.read_csv('D:\SubwayData\sj201412.txt',sep='\t') #加载papa.txt,指定它的分隔符是 \t
papa.head(10) #显示数据的前几行
#s = pd.to_datetime(['20160103 12:34:56','20150203 23:34:45','20150203 23:34:45','20150203 23:34:45','20150203 23:34:45','20150203 23:34:45'])
#print s