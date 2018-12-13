import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

def getFlowMainProcess(dts):
    for dt in dts:
        wt = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\WaitTime\waitTime_for14_line1_' + dt + '.csv')
        x = []
        y = []
        for i in range(1, 205):
            check_seconds = i * 300
            x.append(check_seconds)
            flow = wt[(wt.in_seconds > (check_seconds - 300)) & (wt.in_seconds <= check_seconds)].shape[0]
            y.append(flow)
        fl = pd.DataFrame({'check_seconds': x,
                            'flow': y},
                           columns=['check_seconds', 'flow'])
        fl.to_csv('E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_' + dt + '.csv')

dts = ['20141201', '20141202', '20141203', '20141204', '20141205', '20141206', '20141207'
    , '20141208', '20141209', '20141210', '20141211', '20141212', '20141213', '20141214'
    , '20141215', '20141216', '20141217', '20141218', '20141219', '20141220', '20141221'
    , '20141223', '20141225', '20141226', '20141227', '20141228', '20141229']
getFlowMainProcess(dts)

temp = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_20141229.csv')
plt.plot(temp.check_seconds, temp.flow)
plt.show()