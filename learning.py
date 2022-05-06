import os
import inflect
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

import torch.cuda
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model import *

if torch.cuda.is_available():
    device_name = 'cuda'
    print('Use GPU')
else:
    device_name = 'cpu'
    print('Use CPU')
device = torch.device(device_name)

epochs = 150
lr = 1e-4
batch_size = 32
seq_len = 252
num_heads = 4
num_cols_rfe = 21
p = inflect.engine()


def RFE(X, y):
    df_x = pd.DataFrame(X)
    while len(df_x.columns) > num_cols_rfe:
        RF_model = RandomForestClassifier(max_depth=20, max_features=df_x//10, n_estimators=100)
        RF_model.fit(df_x, y)
        fimp = RF_model.feature_importances_
        to_drop_col = pd.Series(fimp, index=df_x.columns).sort_values().index[:2]
        df_x.drop([to_drop_col], axis=1, inplace=True)
    return df_x.columns


def data_loader(data, label, batch_size, seq_len):
    x = []
    y = []
    i = seq_len
    while i < len(data):
        x.append(data[i-seq_len:i])
        y.append(label.values[i])
        i += 1
    x_tensor = torch.tensor(np.array(x, dtype=np.float32)).to(device)
    y_tensor = torch.tensor(np.array(y, dtype=np.float32).reshape(-1,1)).to(device)
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader


def realtime_test(X, label, ret, is_quantile):
    if is_quantile:
        criterion = 0.
        y = ret.copy()
        loss_fn = nn.MSELoss()
    else:
        criterion = 0.5
        y = label.copy().astype(np.int)
        loss_fn = nn.BCELoss()
    result = pd.DataFrame(columns=['pred', 'label'])

    month_check = 1
    rfe_cols = X.columns
    train_end_date = pd.Timestamp(year=2017, month=1, day=1)
    test_date = train_end_date + pd.DateOffset(months=1)
    train_end_idx = sum(X.index < train_end_date)
    test_idx = sum(X.index < test_date)
    while test_date < X.index[-1] - pd.DateOffset(weeks=1):
        if month_check != train_end_date.month:
            print(train_end_date)
            month_check = train_end_date.month
            rfe_cols = RFE(X[:train_end_idx], label[:train_end_idx].values.ravel())

        train_x = X[rfe_cols][:train_end_idx]
        mms = MinMaxScaler((-1, 1))
        train_x = mms.fit_transform(train_x)
        train_y = y[:train_end_idx]
        train_loader = data_loader(train_x, train_y, batch_size, seq_len)
        test_x = mms.transform(X[rfe_cols][test_idx - seq_len:test_idx])
        test_x = torch.tensor(test_x.reshape(-1, seq_len, len(X.columns)), dtype=torch.float32).to(device)
        test_y = y.iloc[test_idx]

        net = SimpleModel(len(rfe_cols), seq_len, is_quantile).to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        for epoch in range(epochs):
            tloss = 0.0
            tcorrect = 0.0
            total_len = 0.0
            net.train()
            len_loader = 0
            for i, tdata in enumerate(train_loader, 1):
                train_data, train_label = tdata
                if train_data.shape[0] == 1:
                    continue
                optimizer.zero_grad()
                outputs = net(train_data)
                loss = loss_fn(outputs, train_label)
                loss.backward()
                optimizer.step()
                tloss += loss.item()
                outputs[outputs > criterion] = 1.0
                outputs[outputs < criterion] = 0.0
                label_copied = train_label.clone().detach()
                tcorrect += torch.sum(outputs == label_copied).item()
                total_len += len(train_label)
                len_loader = i
            avg_loss = tloss / len_loader
            tcorrect /= total_len
            print(f'Epoch {epoch + 1}  accuracy {round(tcorrect, 4)}  loss {round(avg_loss, 4)}')
            if tcorrect > 0.8:
                break
        net.eval()
        result.loc[X.index[test_idx]] = [net(test_x).detach().item(), test_y]

        train_end_date += pd.DateOffset(weeks=1)
        test_date = train_end_date + pd.DateOffset(months=1)
        train_end_idx = sum(X.index < train_end_date)
        test_idx = sum(X.index < test_date)

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