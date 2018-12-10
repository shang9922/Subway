#coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

#star_time = datetime.datetime.strptime('20141202' + '073000', '%Y%m%d%H%M%S')
#end_time = datetime.datetime.strptime('20141202' + '090000', '%Y%m%d%H%M%S')


print 'Start reading data...'
df = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_20141201.csv")
df2 = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_20141202.csv")
df3 = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_20141203.csv")
df4 = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_20141204.csv")
df5 = pd.read_csv("E:\Pycharm\PythonProjects\Subway\data\TrainData\TrainData_for14_line1_20141205.csv")
print 'Data has been read yet.'

x = []
y = []

x = df.train_num
y1 = df.left_num/df.total_num
y2 = df2.board_num/df2.total_num
y3 = df3.board_num/df3.total_num
y4 = df4.left_num/df4.total_num
y5 = df5.left_num/df5.arr_num

#y2 = df.left_num
#y3 = df.arr_num/df.duration
#y3 = df.arr_num
#print np.mean(df.arr_num)

plt.plot(x, y1, c = 'b', label = '1')
#plt.plot(x, y2, c = 'r', label = '2')
#plt.plot(x, y3, c = 'g', label = '3')
plt.plot(x, y4, c = 'c', label = '4')
#plt.plot(x, y5, c = 'k', label = '5')
plt.legend()
plt.show()