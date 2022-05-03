import os
import inflect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
epochs = 50
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


def realtime_test(X, ret, label, bm, args):
    test_label = []
    y_pred = []
    dates = []
    if args.quantile:
        y = ret.copy()
        loss_fn = nn.MSELoss()
        criterion = 0.
    else:
        y = label.copy()
        loss_fn = nn.BCELoss()
        criterion = 0.5

    filepath = 'result/learning.csv'
    month_check = 1
    if os.path.exists(filepath):
        last_result = pd.read_csv(filepath, index_col='date', parse_dates=True)
        train_data_end_date = last_result.index[-1] + pd.DateOffset(weeks=1) - pd.DateOffset(months=1)
        month_check = train_data_end_date.month
    else:
        last_result = pd.DataFrame()
        train_data_end_date = pd.Timestamp(year=2016, month=1, day=1)
    test_date = train_data_end_date + pd.DateOffset(months=1)

    while test_date < X.index[-1] - pd.DateOffset(months=1):
        # 1달마다 저장
        if month_check != train_data_end_date.month:
            print(train_data_end_date)
            month_check = train_data_end_date.month
            result = pd.DataFrame({'date':dates, 'pred_ret':y_pred, 'test_label':test_label})
            result = pd.concat([last_result, result.set_index('date')])
            result.to_csv(filepath)
        train_data_end_idx = sum(X.index < train_data_end_date)
        test_idx = sum(X.index < test_date) + 1

        if args.corr:
            bm_related_cols = correlation(bm, X, train_data_end_idx, corr_term)
        else:
            bm_related_cols = X.columns
        train_x = X[bm_related_cols][:train_data_end_idx]
        mms = MinMaxScaler((-1,1))
        train_x = mms.fit_transform(train_x)
        train_y = y[:train_data_end_idx]
        train_loader = data_loader(train_x, train_y, batch_size, seq_len)
        test_x = mms.transform(X[bm_related_cols][test_idx-seq_len:test_idx])
        test_x = torch.tensor(test_x.reshape(-1, seq_len, len(bm_related_cols)), dtype=torch.float32).to(device)
        test_y = label.iloc[test_idx]

        net = MyModel(seq_len, len(bm_related_cols), num_heads, args.ts_layer, args.quantile).to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        for epoch in range(epochs):
            tloss = 0.0
            tcorrect = 0.0
            total_len = 0.0
            net.train()
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
                if args.quantile:
                    label_copied[label_copied > criterion] = 1.0
                    label_copied[label_copied < criterion] = 0.0
                tcorrect += torch.sum(outputs == label_copied).item()
                total_len += len(train_label)
            avg_loss = tloss / i
            tcorrect /= total_len
            print(f'Epoch {epoch+1}  accuracy {round(tcorrect, 4)}  loss {round(avg_loss, 4)}')
        net.eval()
        y_pred.append(net(test_x).detach().item())
        test_label.append(test_y)
        dates.append(X.index[test_idx])

        train_data_end_date += pd.DateOffset(weeks=1)
        test_date = train_data_end_date + pd.DateOffset(months=1)

    return y_pred, test_label


def purged_kfold(X, ret, label, num_fold, ts_layer, is_quantile):
    taccus = []
    vaccus = []
    if is_quantile:
        y = ret.copy()
    else:
        y = label.copy()

    # transformer 적용시 num_cols % num_heads == 0 필수 -> 일부 column 제외
    num_cols = len(X.columns)
    while num_cols % num_heads != 0:
        num_cols -= 1
    X = X[X.columns[:num_cols]]

    val_len = len(X) // num_fold
    val_end_idx = len(X) - 1
    val_start_idx = val_end_idx - val_len
    val_start_date = X.index[val_start_idx]
    train1_end_date = val_start_date - pd.DateOffset(months=1)
    train1_end_idx = sum(X.index < train1_end_date)
    train2_start_date, train2_start_idx = None, None
    for i in range(num_fold):
        train_x = pd.DataFrame()
        train_y = pd.Series()
        if train1_end_idx is not None:
            train_x = pd.concat([train_x, X[:train1_end_idx]])
            train_y = pd.concat([train_y, y[:train1_end_idx]])
        if train2_start_idx is not None:
            train_x = pd.concat([train_x, X[train2_start_idx:]])
            train_y = pd.concat([train_y, y[train2_start_idx:]])
        val_x = X[val_start_idx:val_end_idx]
        val_y = y[val_start_idx:val_end_idx]

        print(f'{p.ordinal(i+1)} fold learning')
        taccu, vaccu = kfold_test(train_x, train_y, val_x, val_y, ts_layer, is_quantile, i+1)
        taccus.append(taccu)
        vaccus.append(vaccu)
        print()

        # date change
        val_end_idx -= val_len
        val_start_idx = val_end_idx - val_len
        val_start_date = X.index[val_start_idx]
        train2_start_idx = val_start_idx + seq_len
        train2_start_date = X.index[train2_start_idx] + pd.DateOffset(months=1)
        train2_start_idx = sum(X.index < train2_start_date)
        if i+1 == num_fold:
            train1_end_idx = None
        else:
            train1_end_date = val_start_date - pd.DateOffset(months=1)
            train1_end_idx = sum(X.index < train1_end_date)
    print(f'Final Result  taccu {round(sum(taccus) * 100 / num_fold, 4)}  vaccu {round(sum(vaccus) * 100 / num_fold, 4)}')
    return


def kfold_test(train_x, train_y, val_x, val_y, ts_layer, is_quantile, nth_fold):
    tlosses = []
    vlosses = []
    if is_quantile:
        # loss_fn = quantile_loss
        loss_fn = nn.MSELoss()
        criterion = 0.
    else:
        loss_fn = nn.BCELoss()
        criterion = 0.5

    mms = MinMaxScaler((-1,1))
    train_x = mms.fit_transform(train_x)
    train_loader = data_loader(train_x, train_y, batch_size, seq_len)
    val_x = mms.transform(val_x)
    val_loader = data_loader(val_x, val_y, batch_size, seq_len)

    net = MyModel(seq_len, train_x.shape[-1], num_heads, ts_layer, is_quantile)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 100)
    taccu, vaccu = 0., 0.
    min_val_loss = 1e5
    min_val_accu = 0.
    early_stop_cnt = 0
    for epoch in range(epochs):
        tloss = 0.0
        vloss = 0.0
        taccu = 0.0
        vaccu = 0.0
        total_len = 0.0
        net.train()
        for i, tdata in enumerate(train_loader, 1):
            train_data, train_label = tdata
            optimizer.zero_grad()
            outputs = net(train_data)
            loss = loss_fn(outputs, train_label)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            outputs[outputs > criterion] = 1.0
            outputs[outputs < criterion] = 0.0
            label_copied = train_label.clone().detach()
            if is_quantile:
                label_copied[label_copied > criterion] = 1.0
                label_copied[label_copied < criterion] = 0.0
            taccu += torch.sum(outputs == label_copied).item()
            total_len += len(train_label)
        avg_loss = tloss / i
        taccu /= total_len

        total_len = 0.0
        net.eval()
        for i, vdata in enumerate(val_loader, 1):
            val_data, val_label = vdata
            outputs = net(val_data)
            loss = loss_fn(outputs, val_label)
            vloss += loss.item()
            outputs[outputs > criterion] = 1.0
            outputs[outputs < criterion] = 0.0
            label_copied = val_label.clone().detach()
            if is_quantile:
                label_copied[label_copied > criterion] = 1.0
                label_copied[label_copied < criterion] = 0.0
            vaccu += torch.sum(outputs == label_copied).item()
            total_len += len(val_label)
        val_avg_loss = vloss / i
        vaccu /= total_len
        print(f'Epoch {epoch+1}  accuracy {round(taccu*100, 4)}  loss {round(avg_loss, 4)}  '\
        f'val accuracy {round(vaccu*100, 4)}  val loss {round(val_avg_loss, 4)}')
        tlosses.append(round(avg_loss, 4))
        vlosses.append(round(val_avg_loss, 4))
        scheduler.step()

        # early stopping
        if val_avg_loss < min_val_loss:
            min_val_loss = val_avg_loss
            min_val_accu = vaccu
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt > 30:
            break
    plt.plot(tlosses, label='train')
    plt.plot(vlosses, label='val')
    plt.legend()
    plt.savefig(f'result/fold{nth_fold}.png')
    plt.clf()
    return taccu, min_val_accu