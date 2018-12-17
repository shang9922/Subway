# coding:utf-8
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
    , '20141208', '20141209', '20141210', '20141211', '20141212'
    , '20141215', '20141216', '20141217', '20141218', '20141219'
    , '20141223', '20141225', '20141226']

train_df = pd.DataFrame(columns=['train_num', 'check_seconds', 'duration'])
print('Combining data ...')
j = 1
for date_str in dts:
    temp_df = pd.read_csv(
        'E:\Pycharm\PythonProjects\Subway\data\TimeTable\TimeTable_for14_line1_' + date_str + '.csv')
    duration = []
    for i in range(temp_df.shape[0]):
        temp_df.iloc[i, 0] = i + 1 + j*0.025
        if i<1:
            duration.append(0)
        else:
            duration.append(temp_df.iloc[i, 1] - temp_df.iloc[i-1, 1])
    temp_df['duration'] = duration
    temp_df.columns = ['train_num', 'check_seconds', 'duration']
    train_df = train_df.append(temp_df, ignore_index=True)
    j = j+1
# plt.scatter(train_df[train_df.train_num<=20].train_num, train_df[train_df.train_num<=20].duration, marker='.', color='k', s=10)
plt.scatter(train_df[train_df.train_num <= 10].train_num, train_df[train_df.train_num <= 10].check_seconds, marker='.', color='k', s=10)
plt.show()
