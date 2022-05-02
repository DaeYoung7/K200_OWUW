import os
import inflect
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

epochs = 150
lr = 1e-4
batch_size = 32
seq_len = 252
num_heads = 4
corr_term = 252 * 1
p = inflect.engine()


def realtime_test(X, y, bm):
    cnt = 0
    pd.DataFrame(columns=list(X.columns)+['date', 'LR', 'SVM', 'label'])

    year_check = 1
    train_end_date = pd.Timestamp(year=2012, month=1, day=1)
    next_train_end_date = train_end_date + pd.DateOffset(weeks=1)
    test_date = train_end_date + pd.DateOffset(months=1)

    while test_date < X.index[-1] - pd.DateOffset(months=1):
        # 1달마다 저장
        if year_check != train_end_date.month:
            print(train_end_date)
            year_check = train_end_date.month
        train_end_idx = sum(X.index < train_end_date)
        test_idx = sum(X.index < test_date) + 1

        train_x = X[:train_end_idx]
        mms = MinMaxScaler((0,1))
        train_x = mms.fit_transform(train_x)
        train_y = y[:train_end_idx]
        test_x = mms.transform(X.iloc[test_idx].values.reshape((1,-1)))
        test_y = y.iloc[test_idx]

        LR_model = LogisticRegression(C=0.1, penalty='l1', solver='saga')
        LR_model.fit(train_x, train_y)
        lr_pred = LR_model.predict(test_x)

        LR_model = SVC(C=0.1, penalty='l1', solver='saga')
        LR_model.fit(train_x, train_y)
        lr_pred = LR_model.predict(test_x)

        test_label.append(test_y)
        dates.append(X.index[test_idx])

        train_data_end_date += pd.DateOffset(days=1)
        test_date = train_data_end_date + pd.DateOffset(months=1)
        cnt += 1
        if cnt > 50:
            break
    return y_pred, test_label, dates
