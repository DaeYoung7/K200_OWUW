import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self, seq_len, num_feature, num_heads, ts_layer, is_quantile):
        super().__init__()
        self.seq_len = seq_len
        self.is_quantile = is_quantile
        self.ts_layer_name = ts_layer
        # layers
        w_hidden_size = 256 if ts_layer == 'lstm' else num_feature
        cnn_hidden_size = 256
        cnn_kernel_size1 = 64
        cnn_kernel_size2 = 16
        cnn_kernel_size3 = 3
        pooling_size = 3
        size_after_cnn = cnn_hidden_size - cnn_kernel_size1 - cnn_kernel_size2 - cnn_kernel_size3 + 3 - pooling_size * 3 + 3
        ff_hidden_size = 64
        self.layer_norm = nn.LayerNorm([seq_len, num_feature])
        if ts_layer == 'lstm':
            self.ts_layer = nn.LSTM(num_feature, w_hidden_size, 2)
        elif ts_layer == 'transformer':
            encoder_layer = nn.TransformerEncoderLayer(num_feature, num_heads, 512)
            self.ts_layer = nn.TransformerEncoder(encoder_layer, 3)
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
        # ts_hidden_state - (batch_size, seq_len, w_hidden_size)
        ts_hidden_state = None
        if self.ts_layer_name == 'lstm':
            ts_hidden_state, _ = self.ts_layer(self.layer_norm(inputs))
        elif self.ts_layer_name == 'transformer':
            ts_hidden_state = self.ts_layer(self.layer_norm(inputs))

        # last_ht - (batch_size, w_hidden_size)
        last_ht = ts_hidden_state[:, -1, :]
        last_ht = torch.squeeze(last_ht, 1)
        hidden_state = torch.relu(self.weight(self.batch_norm1(ts_hidden_state)))

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

