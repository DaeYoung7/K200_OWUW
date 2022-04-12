import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, seq_len, num_feature, is_quantile):
        super().__init__()
        self.seq_len = seq_len
        self.is_quantile = is_quantile

        # layers
        w_hidden_size = 256
        cnn_hidden_size = 256
        cnn_kernel_size1 = 64
        cnn_kernel_size2 = 16
        cnn_kernel_size3 = 3
        pooling_size = 3
        size_after_cnn = cnn_hidden_size - cnn_kernel_size1 - cnn_kernel_size2 - cnn_kernel_size3 + 3 - pooling_size * 3 + 3
        ff_hidden_size = 64
        self.layer_norm = nn.LayerNorm([seq_len, num_feature])
        self.lstm = nn.LSTM(num_feature, w_hidden_size, 2)
        self.batch_norm1 = nn.BatchNorm1d(seq_len)
        self.weight = nn.Linear(w_hidden_size, cnn_hidden_size, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(seq_len)
        self.cnn1 = nn.Conv1d(self.seq_len, self.seq_len, kernel_size=cnn_kernel_size1)
        self.batch_norm3 = nn.BatchNorm1d(seq_len)
        self.cnn2 = nn.Conv1d(self.seq_len, self.seq_len, kernel_size=cnn_kernel_size2)
        self.batch_norm4 = nn.BatchNorm1d(seq_len)
        self.cnn3 = nn.Conv1d(self.seq_len, self.seq_len, kernel_size=cnn_kernel_size3)
        self.batch_norm5 = nn.BatchNorm1d(w_hidden_size+size_after_cnn)
        self.linear1 = nn.Linear(w_hidden_size+size_after_cnn, ff_hidden_size, bias=False)
        self.batch_norm6 = nn.BatchNorm1d(ff_hidden_size)
        self.linear2 = nn.Linear(ff_hidden_size, 1, bias=False)

        self.maxpool = nn.MaxPool1d(3, stride=1)

    def forward(self, inputs):
        # hidden_state - (batch_size, seq_len, hidden_size1)
        lstm_hidden_state, _ = self.lstm(self.layer_norm(inputs))
        # last_ht - (batch_size, hidden_size1)
        last_ht = lstm_hidden_state[:, -1, :]
        last_ht = torch.squeeze(last_ht, 1)
        hidden_state = torch.relu(self.weight(self.batch_norm1(lstm_hidden_state)))

        # cnn_ht1 - (batch_size, seq_len, w_hidden_size - cnn_kernel_size1 + 1 - pooling_size + 1)
        cnn_ht1 = self.maxpool(self.cnn1(self.batch_norm2(hidden_state)))
        cnn_ht2 = self.maxpool(self.cnn2(self.batch_norm3(cnn_ht1)))
        # cnn_ht3 - (batch_size, seq_len, size_after_cnn)
        cnn_ht3 = self.maxpool(self.cnn3(self.batch_norm4(cnn_ht2)))

        # s - (batch_size, seq_len)
        s = torch.tanh(torch.mean(cnn_ht3, -1))
        # a - (batch_size, size_after_cnn)
        a = torch.mean(torch.mul(s.reshape([-1, self.seq_len, 1]), cnn_ht3), 1)
        # v - (batch_size, w_hidden_size + size_after_cnn)
        v = torch.cat([last_ht, a], -1)
        # c - (batch_size, ff_hidden_size)
        c = torch.relu(self.linear1(self.batch_norm5(v)))
        if self.is_quantile:
            outputs = self.linear2(self.batch_norm6(c))
        else:
            outputs = torch.sigmoid(self.linear2(self.batch_norm6(c)))
        return outputs


q = 0.5
def quantile_loss(preds, target):
    errors = target - preds
    losses = torch.max((q - 1) * errors, q * errors)
    loss = torch.mean(torch.sum(losses, dim=1))
    return loss

