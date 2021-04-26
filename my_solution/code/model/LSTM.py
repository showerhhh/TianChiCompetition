import torch
import torch.nn.functional as F
from torch import nn

import config as cnf


class LSTM_MODULE(nn.Module):
    def __init__(self):
        super(LSTM_MODULE, self).__init__()
        self.lstm = nn.LSTM(input_size=cnf.embed_dim, hidden_size=cnf.hidden_size, num_layers=cnf.num_layers,
                            batch_first=True, bidirectional=True, dropout=cnf.dropout)
        self.fc1 = nn.Linear(cnf.hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, cnf.output_size)

    def forward(self, x):
        """
        :param input: 形状[batch_size,max_len],其中max_len表示每个句子有多少单词
        :return:
        """
        # x = self.embedding(input)   # 输出形状:[batch_size,seq_len,embedding_dim]
        # 经过lstm层，x:[batch_size,max_len,2*hidden_size]
        # h_n,c_n:[2*num_layers,batch_size,hidden_size]
        x, (h_n, c_n) = self.lstm(x)

        # 获取两个方向最后一次的output,进行concat
        output_fw = h_n[-2, :, :]  # 正向最后一次输出
        output_bw = h_n[-1, :, :]  # 反向最后一次输出

        output = torch.cat([output_fw, output_bw], dim=-1)

        out_fc1 = self.fc1(output)
        out_relu = F.relu(out_fc1)
        out = self.fc2(out_relu)
        return out


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.extractor = LSTM_MODULE()
        self.fc1 = nn.Linear(2 * cnf.output_size, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, q1, q2):
        y1 = self.extractor(q1)  # (batch, output_size)
        y2 = self.extractor(q2)  # (batch, output_size)
        y1_y2 = torch.cat([y1, y2], dim=1)  # (batch, 2*output_size)
        out = self.fc2(F.relu(self.fc1(y1_y2)))  # (batch, 2)
        return out
