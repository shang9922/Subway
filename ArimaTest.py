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
    train_dt = ['20141201', '20141202', '20141203', '20141204'
        , '20141208', '20141209', '20141210', '20141211'
        , '20141215', '20141216', '20141217', '20141218'
                , '20141223', '20141225', '20141226']
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

    decomposition = seasonal_decompose(ts, freq=predict_len, two_sided=False, model='additive')
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    # residual.dropna(inplace=True)
    # decomposition.plot()
    # plt.show()
    trend.dropna(inplace=True)
    # print testStationarity(trend)
    trend_anl = trend.diff(periods=1).dropna()
    # trend_anl.plot()
    # plt.show()
    # print testStationarity(trend_anl)
    # trend_anl = trend.diff().dropna().diff().dropna()
    # draw_acf_pacf(trend_anl)
    # draw_acf_pacf(trend_anl)
    # p_and_q(trend)
    # trend_model = ARIMA(trend, order=(4, 0, 0))
    trend_model = ARIMA(trend, order=(2, 1, 2))
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
    plt.legend()
    plt.show()
    print r2_score(test_df.flow, pre_rs)
    print mean_absolute_error(pre_rs, test_df.flow)
    print mean_squared_error(pre_rs, test_df.flow)
    return pre_rs


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
    plot_acf(ts, lags=40, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=40, ax=ax2)
    plt.show()

def testStationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput

def p_and_q(trend):
    pmax = 6
    qmax = 6
    d=1
    aic_matrix = []  # bic矩阵
    bic_matrix = []  # bic矩阵
    aic_lest = 0
    bic_lest = 0
    p_aic = 0
    q_aic = 0
    p_bic = 0
    q_bic = 0
    for p in range(pmax + 1):
        aic_tmp = []
        bic_tmp = []
        for q in range(qmax + 1):
            try:
                a = ARIMA(trend, (p, d, q)).fit().aic
                b = ARIMA(trend, (p, d, q)).fit().bic
                if a<aic_lest:
                    aic_lest = a
                    p_aic = p
                    q_aic = q
                if b<bic_lest:
                    bic_lest = b
                    p_bic = p
                    q_bic = q
                aic_tmp.append(a)
                bic_tmp.append(b)
            except:
                aic_tmp.append(0)
                bic_tmp.append(0)
        aic_matrix.append(aic_tmp)
        bic_matrix.append(bic_tmp)
    aic_df = pd.DataFrame(aic_matrix)
    bic_df = pd.DataFrame(bic_matrix)
    aic_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\Aic_matrix_26.csv')
    bic_df.to_csv('E:\Pycharm\PythonProjects\Subway\data\TrainData\Bic_matrix_26.csv')
    print 'aic: p = %d, q = %d', (p_aic, q_aic)
    print 'bic: p = %d, q = %d', (p_bic, q_bic)
    print 'finish'


# dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
#     , '20141208', '20141209', '20141210', '20141211', '20141212'
#     , '20141215', '20141216', '20141217', '20141218', '20141219']
# # dts = ['20141201', '20141202', '20141203', '20141204', '20141205']
# arimaMainProcess(dts, 204, '20141229')

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
if __name__ == '__main__':
    dts = ['20141201', '20141202', '20141203', '20141204', '20141205'
        , '20141208', '20141209', '20141210', '20141211', '20141212'
        , '20141215', '20141216', '20141217', '20141218', '20141219']
    arimaMainProcess(dts, 204, '20141229')