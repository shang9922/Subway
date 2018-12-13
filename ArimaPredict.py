# coding:utf-8
import matplotlib.pyplot as plt
import pandas as pd
from pyramid.arima import auto_arima
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def arimaMainProcess(train_dt, predict_len, test_str):
    train_df = pd.DataFrame(columns=['check_seconds', 'flow'])
    print('Combining data ...')
    for date_str in train_dt:
        temp_df = pd.read_csv(
            'E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_' + date_str + '.csv')
        train_df = train_df.append(temp_df, ignore_index=True)

    len = train_df.shape[0]
    train_df = train_df.loc[:, ['flow']]
    train_df = pd.DataFrame(train_df.flow.values, index=pd.date_range('2008-01-01', periods=len), columns=['flow'])
    train_df.index = pd.to_datetime(train_df.index)
    train_df['flow'] = train_df['flow'].astype(float)
    ts = train_df['flow']
    #print testStationarity(ts)


    decomposition = seasonal_decompose(ts, freq=predict_len, two_sided=False)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    residual.dropna(inplace=True)
    #print testStationarity(residual)
    #draw_acf_pacf(residual)
    #decomposition.plot()
    #plt.show()
    trend.dropna(inplace=True)
    # trend_anl = trend.diff().dropna()
    # draw_acf_pacf(trend_anl)
    trend_model = ARIMA(trend, order=(0, 1, 2))
    trend_arma = trend_model.fit(disp=0)
    trend_rs = trend_arma.forecast(predict_len)[0]
    pre_rs = []
    train_terms = ts.shape[0]/predict_len
    for i in range(predict_len):
        temp = []
        for j in range(train_terms):
            temp.append(seasonal[j*predict_len+i])
        seasonal_part = np.mean(temp)
        pre_rs.append(trend_rs[i]+seasonal_part)

    pre_rs = [round(i) for i in pre_rs]
    test_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_' + test_str + '.csv')
    plt.plot(test_df.check_seconds, pre_rs, c='r', label='arima')
    plt.plot(test_df.check_seconds, test_df.flow, c='b', label='real data')
    plt.show()
    print mean_absolute_error(pre_rs, test_df.flow)
    print mean_squared_error(pre_rs, test_df.flow)


    # differenced = difference(ts, 204)
    #print differenced
    #differenced = pd.Series(differenced).diff().dropna()
    # print testStationarity(differenced)
    # draw_acf_pacf(differenced)
    # differenced = pd.Series(differenced).diff().dropna()
    # print testStationarity(differenced)
    # draw_acf_pacf(differenced)
    # model = ARIMA(differenced, order=(1, 0, 1))
    # result_arma = model.fit(disp=0)
    # rs = result_arma.forecast(204)[0]
    # print rs
    # history = [x for x in ts]
    # new_rs = []
    # for yhat in rs:
    #     inverted = inverse_difference(history, yhat, 204)
    #     print('Flow : %f' % (inverted))
    #     history.append(inverted)
    #     new_rs.append(inverted)
    # # print rs
    # #print testStationarity(dif)
    # test_df = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\InFlow\InFlow_for14_line1_20141225.csv')
    # plt.plot(test_df.check_seconds, new_rs, c='r', label='arima')
    # plt.plot(test_df.check_seconds, test_df.flow, c='b', label='real data')
    # plt.show()
    # print mean_absolute_error(new_rs, test_df.flow)
    # print mean_squared_error(new_rs, test_df.flow)



def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

def draw_ts(timeseries):
    timeseries.plot()
    plt.show()

def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4],index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput

dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
    , '20141208', '20141209', '20141210', '20141211', '20141212'
    , '20141215', '20141216', '20141217', '20141218', '20141219'
    , '20141223', '20141225', '20141226']
# dts = ['20141201', '20141202', '20141203', '20141204', '20141205']
arimaMainProcess(dts, 204, '20141229')

# df2 = pd.read_csv('E:\Pycharm\PythonProjects\Subway\data\WaitTime\waitTime_for14_line1_20141203.csv')
# x = []
# y = []
# for i in range(1, 205):
#     check_seconds = i * 300
#     x.append(check_seconds)
#     count = df2[(df2.in_seconds > (check_seconds - 300)) & (df2.in_seconds <= check_seconds)].shape[0]
#     y.append(count)
# plt.plot(x, y)
# plt.show()
