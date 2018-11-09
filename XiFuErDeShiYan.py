#coding:utf-8
import datetime
import pandas as pd

writeFile = open("C:\Users\peng\Desktop\szw.txt","w")
df = pd.read_excel('C:\Users\peng\Desktop\\report.xls',usecols=[9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39])
df = df.iloc[11:41,:]
length = df.shape[0]
w = df.shape[1]
data = []

for x in range(length):
    for y in range(w):
        data.append('6/%d/2002 %d:00 %d\n' % (x+1, y+5, df.iloc[x][y]))

writeFile.writelines(data)

writeFile.close()