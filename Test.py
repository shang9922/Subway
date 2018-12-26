# coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

# test_dt = '20141229'
#
#
# test = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\PlatformCount\PlatformCount_for_' + test_dt + '.csv').loc[:, ['check_seconds', 'count']]
# temp = []
# for i in test.check_seconds:
#     temp.append((datetime.datetime.strptime(test_dt + '063000', '%Y%m%d%H%M%S') + datetime.timedelta(seconds=int(i))).time())
# plt.plot(temp[380:720], test['count'][380:720], c='b')
# plt.title('Passenger Number on the Platform')
# plt.xlabel('Time')
# plt.ylabel('Passenger Number')
# plt.xticks(['08:00', '08:30', '09:00', '09:30'])
# plt.show()

y1 = [8.17, 8.95, 14.20]
y2 = [4.86, 7.65, 12.23]
index = np.arange(3)

bar_width = 0.3
plt.bar(index+1-0.15, y1, width=0.3, color='y', label='RMSE')
plt.bar(index+1+0.15, y2, width=0.3, color='b', label='MAE')
plt.legend()
plt.xticks([1,2,3],('SVRBasePre','AR','HA'))
plt.show()