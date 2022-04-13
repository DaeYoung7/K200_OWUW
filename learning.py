import numpy as np
import pandas as pd
import torch.cuda
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from model import *
device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

def correlation(bm, data, date_idx, term):
    corr = pd.concat([bm[date_idx-term:date_idx], data[date_idx-term:date_idx]], axis=1).corr()
    corr_abs = abs(corr.iloc[0]).sort_values(ascending=False)
    num_cols = len(corr_abs[corr_abs > 0.3].index[1:])
    while num_cols % 4 !=0:
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


epochs = 50
batch_size = 32
seq_len = 252
corr_term = 252 * 1


def train_test(X, ret, label, bm, ts_layer, is_quantile):
    test_label = []
    y_pred = []
    if is_quantile:
        y = ret.copy()
        loss_fn = nn.L1Loss()
        criterion = 0.
    else:
        y = label.copy()
        loss_fn = nn.BCELoss()
        criterion = 0.5
    # 상관관계를 파악할 기간 (day)
    train_data_end_date = pd.Timestamp(year=2016, month=1, day=1)
    test_date = train_data_end_date + pd.DateOffset(months=1)
    year_check = 0
    while test_date < X.index[-1] - pd.DateOffset(months=1):
        if year_check != test_date.year:
            year_check = test_date.year
            print(f'- test year {year_check} -')
        train_data_end_idx = sum(X.index < train_data_end_date)
        test_idx = sum(X.index < test_date) + 1

        bm_related_cols = correlation(bm, X, train_data_end_idx, corr_term)
        train_x = X[bm_related_cols][:train_data_end_idx]
        mms = MinMaxScaler((-1,1))
        train_x = mms.fit_transform(train_x)
        train_y = y[:train_data_end_idx]
        train_loader = data_loader(train_x, train_y, batch_size, seq_len)
        test_x = mms.transform(X[bm_related_cols][test_idx-seq_len:test_idx])
        test_x = torch.tensor(test_x.reshape(-1, seq_len, len(bm_related_cols)), dtype=torch.float32)
        test_y = label.iloc[test_idx]

        net = MyModel(seq_len, len(bm_related_cols), ts_layer, is_quantile).to(device)
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        for epoch in range(epochs):
            tloss = 0.0
            tcorrect = 0.0
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
                tcorrect += torch.sum(outputs == label_copied).item()
                total_len += len(train_label)
            avg_loss = tloss / i
            tcorrect /= total_len
            print(f'Epoch {epoch+1}  accuracy {round(tcorrect, 4)}  loss {round(avg_loss, 4)}')
        net.eval()
        y_pred.append(net(test_x).detach().numpy()[0])
        test_label.append(test_y)

        train_data_end_date += pd.DateOffset(weeks=1)
        test_date = train_data_end_date + pd.DateOffset(months=1)
    return y_pred, test_label