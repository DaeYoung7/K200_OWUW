import torch
import torch.nn as nn


class SimpleModel(nn.Module):
    def __init__(self, num_feature, seq_len, is_quantile):
        super().__init__()
        self.is_quantile = is_quantile
        w_hidden_size = [256, 64]
        self.layer_norm = nn.LayerNorm([seq_len, num_feature])
        self.lstm1 = nn.LSTM(num_feature, w_hidden_size[0], 1)
        self.lstm2 = nn.LSTM(w_hidden_size[0], w_hidden_size[1], 1)
        self.batch_norm = nn.BatchNorm1d(w_hidden_size[1])
        self.linear = nn.Linear(64, 1, bias=False)

    def forward(self, inputs):
        x, _ = self.lstm1(self.layer_norm(inputs))
        x, _ = self.lstm2(x)
        last_x = x[:, -1, :]
        last_x = torch.squeeze(last_x, 1)
        ret = self.linear(self.batch_norm(last_x))
        if not self.is_quantile:
            ret = torch.sigmoid(ret)
        return ret
