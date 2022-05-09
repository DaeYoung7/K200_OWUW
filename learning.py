import os
import inflect
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


# version 1
# 개별 모델(앙상블 이전) 학습 -> 학습 데이터에 대한 label생성
# 동일한 학습데이터 + 생성된 label로 앙상블 모델 학습, 백테스트
def direct_ensemble(X, y, bm, is_corr):
    result = pd.DataFrame(columns=['LR', 'SVC', 'RF', 'label'])
    month_check = 1
    train_end_date = pd.Timestamp(year=2012, month=1, day=1)
    test_date = train_end_date + pd.DateOffset(months=1)
    while test_date < X.index[-1] - pd.DateOffset(weeks=1):
        if month_check != train_end_date.month:
            print(train_end_date)
            month_check = train_end_date.month

        train_end_idx = sum(X.index < train_end_date)
        test_idx = sum(X.index < test_date)
        if is_corr:
            bm_related_cols = correlation(bm, X, train_end_idx, corr_term)
        else:
            bm_related_cols = X.columns
        train_x = X[bm_related_cols][:train_end_idx].values
        mms = MinMaxScaler((0, 1))
        train_x = mms.fit_transform(train_x)
        train_y = y[:train_end_idx].values.ravel()
        test_x = mms.transform(X.iloc[test_idx][bm_related_cols].values.reshape((1,-1)))
        test_y = y.iloc[test_idx].values[0]

        # 개별 모델 학습, 학습 데이터에 대한 label 생성
        LR_model = LogisticRegression(C=1e-5, penalty='l1', solver='saga')
        LR_model.fit(train_x, train_y)
        lr_train_pred = LR_model.predict(train_x)
        lr_pred = LR_model.predict(test_x)[0]

        SVC_model = SVC(C=30, gamma=0.001, probability=True)
        SVC_model.fit(train_x, train_y)
        svc_train_pred = SVC_model.predict(train_x)
        svc_pred = SVC_model.predict(test_x)[0]

        train_x_labeled = X[bm_related_cols][:train_end_idx].copy()
        train_x_labeled['LR'] = lr_train_pred
        train_x_labeled['SVC'] = svc_train_pred
        train_x_labeled = train_x_labeled.values

        test_x_labeled = np.append(test_x, np.array([[lr_pred, svc_pred]]), axis=1)

        RF_model = RandomForestClassifier(max_depth=30, max_features=8, n_estimators=100)
        RF_model.fit(train_x_labeled, train_y)
        rf_pred = RF_model.predict(test_x_labeled)[0]

        result.loc[X.index[test_idx]] = [lr_pred, svc_pred, rf_pred, test_y]

        train_end_date += pd.DateOffset(weeks=1)
        test_date = train_end_date + pd.DateOffset(months=1)
    result.index.name = 'date'
    return result

# version 2
# 데이터의 앞부분으로 개별 모델(앙상블 이전)모델 학습 -> 뒷부분에 대한 label 생성
# 뒷부분으로 앙상블 모델 학습, 백테스트
# 상관관계 적용 불가능 (개별 모델과 앙상블 모델의 학습 데이터 기간이 다르기 때문)
def make_label_to_ensemble(X, y):
    result = pd.DataFrame(columns=list(X.columns)+['LR', 'SVC', 'label'])

    month_check = 1
    train_end_date = pd.Timestamp(year=2012, month=1, day=1)
    next_train_end_date = train_end_date + pd.DateOffset(weeks=1)
    test_date = train_end_date + pd.DateOffset(months=1)
    next_test_date = next_train_end_date + pd.DateOffset(months=1)
    while test_date < X.index[-1] - pd.DateOffset(weeks=1):
        if month_check != train_end_date.month:
            print(train_end_date)
            month_check = train_end_date.month
        train_end_idx = sum(X.index < train_end_date)
        test_idx = sum(X.index < test_date)
        next_test_idx = sum(X.index < next_test_date)
        if next_test_idx - test_idx < 1:
            train_end_date = next_train_end_date
            next_train_end_date += pd.DateOffset(weeks=1)
            test_date = next_test_date
            next_test_date = next_train_end_date + pd.DateOffset(months=1)
            continue

        train_x = X[:train_end_idx].values
        mms = MinMaxScaler((0,1))
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