import inflect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model import *

device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
epochs = 50
batch_size = 16
seq_len = 252
corr_term = 252 * 1
num_cnn = 32
p = inflect.engine()


def correlation(bm, data, date_idx, term):
    corr = pd.concat([bm[date_idx-term:date_idx], data[date_idx-term:date_idx]], axis=1).corr()
    corr_abs = abs(corr.iloc[0])
    # bm 제외
    return corr_abs[corr_abs > 0.3].index[1:]


def data_loader(data, label, batch_size, seq_len):
    x = []
    y = []
    i = seq_len
    while i < len(data):
        x.append(data[i-seq_len:i])
        y.append(label.values[i])
        i += 1
    train_x_tensor = torch.tensor(np.array(x, dtype=np.float32)).to(device)
    train_y_tensor = torch.tensor(np.array(y, dtype=np.float32).reshape(-1,1)).to(device)
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    return train_loader

def kfold(X, ret, label, num_fold, is_quantile):
    taccus = []
    vaccus = []
    if is_quantile:
        y = ret.copy()
    else:
        y = label.copy()
    val_len = len(X) // num_fold
    val_end_idx = len(X) - 1
    val_end_date = X.index[val_end_idx]
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
        taccu, vaccu = train_test(train_x, train_y, val_x, val_y, is_quantile, i+1)
        taccus.append(taccu)
        vaccus.append(vaccu)
        print()
        val_end_idx -= val_len
        val_end_date = X.index[val_end_idx]
        val_start_idx = val_end_idx - val_len
        val_start_date = X.index[val_start_idx]
        train2_start_idx = val_start_idx + seq_len
        train2_start_date = X.index[train2_start_idx] + pd.DateOffset(months=1)
        train2_start_idx = sum(X.index < train2_start_date)
        if i+1 == num_fold:
            train1_end_idx = None
            train1_end_date = None
        else:
            train1_end_date = val_start_date - pd.DateOffset(months=1)
            train1_end_idx = sum(X.index < train1_end_date)
    print(f'Final Result  taccu {round(sum(taccus) * 100 / num_fold, 4)}  vaccu {round(sum(vaccus) * 100 / num_fold, 4)}')
    return

def train_test(train_x, train_y, val_x, val_y, is_quantile, nth_fold):
    tlosses = []
    vlosses = []
    if is_quantile:
        loss_fn = quantile_loss
        criterion = 0.
    else:
        loss_fn = nn.BCELoss()
        criterion = 0.5

    mms = MinMaxScaler((-1,1))
    train_x = mms.fit_transform(train_x)
    train_loader = data_loader(train_x, train_y, batch_size, seq_len)
    val_x = mms.transform(val_x)
    val_loader = data_loader(val_x, val_y, batch_size, seq_len)

    net = MyModel(seq_len, train_x.shape[-1], is_quantile)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    taccu, vaccu = 0., 0.
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
    plt.plot(tlosses, label='train')
    plt.plot(vlosses, label='val')
    plt.legend()
    plt.savefig(f'result/fold{nth_fold}.png')
    return taccu, vaccu
