import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, seq_len, num_feature, num_cnn, is_quantile):
        super().__init__()
        self.num_cnn = num_cnn
        self.seq_len = seq_len
        self.is_quantile = is_quantile

        # layers
        hidden_size1 = 256
        hidden_size2 = 256
        hidden_size3 = 64
        self.lstm = nn.LSTM(num_feature, hidden_size1, 2)
        self.cnn_list = nn.ModuleList([nn.Conv1d(self.seq_len, self.seq_len, hidden_size1) for i in range(self.num_cnn)])
        self.linear1 = nn.Linear(num_cnn, hidden_size2)
        self.linear2 = nn.Linear(hidden_size1+hidden_size2, hidden_size3)
        self.linear3 = nn.Linear(hidden_size3, 1)

        # norm
        self.layer_norm1 = nn.LayerNorm([seq_len, num_feature])
        self.batch_norm1 = nn.BatchNorm1d(seq_len)
        self.batch_norm2 = nn.BatchNorm1d(seq_len)
        self.batch_norm3 = nn.BatchNorm1d(hidden_size1+hidden_size2)
        self.batch_norm4 = nn.BatchNorm1d(hidden_size3)


    def forward(self, inputs):
        # hidden_state - (batch_size, seq_len, hidden_size1)
        hidden_state, _ = self.lstm(self.layer_norm1(inputs))
        hidden_state = self.batch_norm1(hidden_state)
        ht = hidden_state[:,-1,:]
        # ht - (batch_size, hidden_size1)
        ht = torch.squeeze(ht, 1)
        attn = []
        for i in range(self.num_cnn):
            attn.append(self.cnn_list[i](hidden_state))

        # attn - (batch_size, seq_len, num_cnn)
        attn = torch.cat(attn, -1)
        # attn - (batch_size, seq_len, hidden_size2)
        attn = torch.relu(self.linear1(self.batch_norm2(attn)))
        # s - (batch_size, seq_len)
        s = torch.tanh(torch.mean(attn, -1))
        # a - (batch_size, hidden_size2)
        a = torch.mean(torch.mul(s.reshape([-1, self.seq_len, 1]), attn), 1)
        # v - (batch_size, hidden_size1 + hidden_size2)
        v = self.batch_norm3(torch.cat([ht, a], -1))
        c = self.batch_norm4(torch.relu(self.linear2(v)))
        if self.is_quantile:
            outputs = self.linear3(c)
        else:
            outputs = torch.sigmoid(self.linear3(c))
        return outputs


q = 0.5
def quantile_loss(preds, target):
    errors = target - preds
    losses = torch.max((q - 1) * errors, q * errors)
    loss = torch.mean(torch.sum(losses, dim=1))
    return loss
