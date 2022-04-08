import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, seq_len, num_feature, num_cnn):
        super().__init__()
        self.num_cnn = num_cnn
        self.seq_len = seq_len
        hidden_size = 128
        self.lstm = nn.LSTM(num_feature, hidden_size, 2)
        self.cnn_list = nn.ModuleList([nn.Conv1d(self.seq_len, self.seq_len, hidden_size) for i in range(self.num_cnn)])
        self.linear = nn.Linear(hidden_size+num_cnn, 1)

    def forward(self, inputs):
        # x - (batch_size, seq_len, hidden_size)
        hidden_state, _ = self.lstm(inputs)
        ht = hidden_state[:,-1,:]
        # ht - (batch_size, hidden_size)
        ht = torch.squeeze(ht, 1)
        attn = []
        for i in range(self.num_cnn):
            attn.append(self.cnn_list[i](hidden_state))

        # attn - (batch_size, seq_len, num_cnn)
        attn = torch.cat(attn, -1)
        # s - (batch_size, seq_len)
        s = torch.sigmoid(torch.mean(attn, -1))
        # a - (batch_size, num_cnn)
        a = torch.mean(torch.mul(s.reshape([-1, self.seq_len, 1]), attn), 1)
        # v - (batch_size, hidden_size + num_cnn)
        v = torch.cat([ht, a], -1)
        outputs = torch.sigmoid(self.linear(v))
        return outputs


q = 0.5
def quantile_loss(preds, target):
    errors = target - preds
    losses = torch.max((q - 1) * errors, q * errors)
    loss = torch.mean(torch.sum(losses, dim=1))
    return loss
