import os
import inflect
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

epochs = 150
lr = 1e-4
batch_size = 32
seq_len = 252
num_heads = 4
num_cols_rfe = 5
p = inflect.engine()


def RFE(X, y):
    df_x = pd.DataFrame(X)
    while len(df_x.columns) < num_cols_rfe:
        RF_model = RandomForestClassifier(max_depth=40, max_features=16, n_estimators=100)
        RF_model.fit(X, y)
        fimp = RF_model.feature_importances_
        print(fimp)
        break
    return


def get_seq_data(data, label, seq_len):
    train_dates = []
    train_x = []
    train_y = []
    test_dates = []
    test_x = []
    test_y = []
    a = defaultdict(dict)

    idx = seq_len
    while idx < len(data):
        train_date = data.index[idx]
        test_date = train_date + pd.DateOffset(months=1)
        test_idx = sum(data.index < test_date)
        if data.index[test_idx] in test_dates:
            continue

        train_dates.append(train_date)
        train_x.append(data[idx-seq_len:idx])
        train_y.append(label.iloc[idx])

        test_dates.append(train_date)
        test_x.append(data[test_idx-seq_len:test_idx])
        test_y.append(data.iloc[test_idx])
        idx += 1
        if idx > 100:
            break
    train_data =


def realtime_test(X, y):
    result = pd.DataFrame(columns=['pred', 'label'])

    month_check = 1
    train_end_date = pd.Timestamp(year=2017, month=1, day=1)
    test_date = train_end_date + pd.DateOffset(months=1)
    while test_date < X.index[-1] - pd.DateOffset(weeks=1):
        if month_check != train_end_date.month:
            print(train_end_date)
            month_check = train_end_date.month
        train_end_idx = sum(X.index < train_end_date)
        test_idx = sum(X.index < test_date)

        train_x = X[:train_end_idx].values
        mms = MinMaxScaler((-1,1))
        train_x = mms.fit_transform(train_x)
        train_y = y[:train_end_idx].values.ravel()
        test_x = mms.transform(X.iloc[test_idx:next_test_idx].values)
        test_y = y[test_idx:next_test_idx]

        LR_model = LogisticRegression(C=1e-5, penalty='l1', solver='saga')
        LR_model.fit(train_x, train_y)
        lr_pred = LR_model.predict(test_x)

        SVC_model = SVC(C=30, gamma=0.001, probability=True)
        SVC_model.fit(train_x, train_y)
        svc_pred = SVC_model.predict(test_x)

        data_to_concat = X.iloc[test_idx:next_test_idx].values.copy()
        data_to_concat = np.append(data_to_concat, np.array([lr_pred, svc_pred, test_y.values.reshape(-1,)]).T, axis=1)
        df_to_concat = pd.DataFrame(data_to_concat, columns=result.columns, index=X.index[test_idx:next_test_idx])
        result = pd.concat([result, df_to_concat])

        train_end_date = next_train_end_date
        next_train_end_date += pd.DateOffset(weeks=1)
        test_date = next_test_date
        next_test_date = next_train_end_date + pd.DateOffset(months=1)

    result.index.name = 'date'
    return result

def ensemble(data):
    y = data['label']
    X = data.drop(['label'], axis=1)

    train_end_date = pd.Timestamp(year=2018, month=1, day=1)
    test_date = train_end_date + pd.DateOffset(months=1)
    month_check = 1

    rf_preds = []
    rf_labels = []
    while test_date < X.index[-1]:
        if month_check != train_end_date.month:
            print(train_end_date)
            month_check = train_end_date.month
        train_end_idx = sum(X.index < train_end_date)
        test_idx = sum(X.index < test_date)
        train_x = X[:train_end_idx].values
        train_y = y[:train_end_idx].values.ravel()
        test_x = X.iloc[test_idx].values.reshape(1,-1)
        test_y = y.iloc[test_idx]
        RF_model = RandomForestClassifier(max_depth=40, max_features=16, n_estimators=100)
        RF_model.fit(train_x, train_y)
        rf_pred = RF_model.predict(test_x)

        rf_preds.append(rf_pred)
        rf_labels.append(test_y)

        train_end_date += pd.DateOffset(weeks=1)
        test_date = train_end_date + pd.DateOffset(months=1)
    return np.array(rf_preds), np.array(rf_labels)