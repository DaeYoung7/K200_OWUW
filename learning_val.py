import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model import *


def correlation(bm, data, date_idx, term):
    corr = pd.concat([bm[date_idx-term:date_idx], data[date_idx-term:date_idx]], axis=1).corr()
    corr_abs = abs(corr.iloc[0])
    # bm 제외
    return corr_abs[corr_abs > 0.3].index[1:]


def data_loader(data, label, batch_size, seq_len):
    train_x = []
    train_y = []
    i = seq_len
    while i < len(data):
        train_x.append(data[i-seq_len:i])
        train_y.append(label.values[i])
        i += 5
    train_x_tensor = torch.tensor(np.array(train_x, dtype=np.float32))
    train_y_tensor = torch.tensor(np.array(train_y, dtype=np.float32).reshape(-1,1))
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_loader


epochs = 20
batch_size = 32
seq_len = 252
corr_term = 252 * 1
num_cnn = 64


def train_test(X, ret, label, bm, is_quantile):
    tlosses = []
    vlosses = []
    if is_quantile:
        y = ret.copy()
        loss_fn = quantile_loss
        criterion = 0.
    else:
        y = label.copy()
        loss_fn = nn.BCELoss()
        criterion = 0.5
    # 상관관계를 파악할 기간 (day)
    train_data_end_date = pd.Timestamp(year=2016, month=1, day=1)
    train_data_end_idx = sum(X.index < train_data_end_date)
    val_start_date = train_data_end_date + pd.DateOffset(months=1)
    val_start_idx = sum(X.index < val_start_date)
    val_end_date = X.index[-1] - pd.DateOffset(months=1)
    vala_end_idx = sum(X.index < val_end_date)

    train_x = X[:train_data_end_idx]
    mms = MinMaxScaler((-1,1))
    train_x = mms.fit_transform(train_x)
    train_y = y[:train_data_end_idx]
    train_loader = data_loader(train_x, train_y, batch_size, seq_len)
    val_x = mms.transform(X[val_start_idx:vala_end_idx])
    val_y = y[val_start_idx:vala_end_idx]
    val_loader = data_loader(val_x, val_y, batch_size, seq_len)

    net = MyModel(seq_len, len(X.columns), num_cnn, is_quantile)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(epochs):
        tloss = 0.0
        vloss = 0.0
        tcorrect = 0.0
        vcorrect = 0.0
        total_len = 0.0
        net.train(True)
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
            tcorrect += torch.sum(outputs == label_copied).item()
            total_len += len(train_label)
        avg_loss = tloss / i
        tcorrect /= total_len

        total_len = 0.0
        net.train(False)
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
            vcorrect += torch.sum(outputs == label_copied).item()
            total_len += len(val_label)
        val_avg_loss = vloss / i
        vcorrect /= total_len
        print(f'Epoch {epoch+1}  accuracy {round(tcorrect, 4)}  loss {round(avg_loss, 4)}  '\
        f'val accuracy {round(vcorrect, 4)}  val loss {round(val_avg_loss, 4)}')
        tlosses.append(round(avg_loss, 4))
        vlosses.append(round(val_avg_loss, 4))
    plt.plot(tlosses, label='train')
    plt.plot(vlosses, label='val')
    plt.show()
    return
