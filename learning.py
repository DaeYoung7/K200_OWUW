import os
import inflect
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

epochs = 150
lr = 1e-4
batch_size = 32
seq_len = 252
num_heads = 4
corr_term = 252 * 1
p = inflect.engine()


def correlation(bm, data, date_idx, term):
    corr = pd.concat([bm[date_idx-term:date_idx], data[date_idx-term:date_idx]], axis=1).corr()
    corr_abs = abs(corr.iloc[0]).sort_values(ascending=False)
    num_cols = len(corr_abs[corr_abs > 0.3].index[1:])
    while num_cols % num_heads !=0:
        num_cols += 1
    # bm 제외
    return corr_abs.index[1:num_cols+1]


def realtime_test(X, y, bm):
    cnt = 0
    test_label = []
    y_pred = []
    dates = []

    month_check = 1
    train_data_end_date = pd.Timestamp(year=2016, month=1, day=1)
    test_date = train_data_end_date + pd.DateOffset(months=1)

    while test_date < X.index[-1] - pd.DateOffset(months=1):
        # 1달마다 저장
        if month_check != train_data_end_date.month:
            print(train_data_end_date)
            month_check = train_data_end_date.month
        train_data_end_idx = sum(X.index < train_data_end_date)
        test_idx = sum(X.index < test_date) + 1

        bm_related_cols = correlation(bm, X, train_data_end_idx, corr_term)
        train_x = X[bm_related_cols][:train_data_end_idx]
        mms = MinMaxScaler((-1,1))
        train_x = mms.fit_transform(train_x)
        train_y = y[:train_data_end_idx]
        test_x = mms.transform(X[bm_related_cols].iloc[test_idx].values.reshape((1,-1)))
        test_y = y.iloc[test_idx]

        model = LogisticRegression(C=0.1, penalty='l1', solver='saga')
        model.fit(train_x, train_y)

        y_pred.append(model.predict(test_x))
        test_label.append(test_y)
        dates.append(X.index[test_idx])

        train_data_end_date += pd.DateOffset(days=1)
        test_date = train_data_end_date + pd.DateOffset(months=1)
        cnt += 1
        if cnt > 50:
            break
    return y_pred, test_label, dates
