import torch
import torch.nn.functional as F
from torch import nn

import config as cnf


class TextCNN_MODULE(nn.Module):
    def __init__(self, filter_num=100, kernel_list=(3, 4, 5)):
        super(TextCNN_MODULE, self).__init__()
        # 三种kernel，size分别是3,4,5，每种kernel有100个
        # 1表示channel_num，filter_num即输出数据通道数，卷积核大小为(kernel, embed_dim)
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(1, filter_num, (kernel, cnf.embed_dim)),
                          nn.LeakyReLU(),
                          nn.MaxPool2d((cnf.max_seq_len - kernel + 1, 1)))
            for kernel in kernel_list
        ])
        self.dropout = nn.Dropout(cnf.dropout)
        self.fc = nn.Linear(filter_num * len(kernel_list), cnf.output_size)

    def forward(self, x):
        # 输入x [128, 10, 100] (batch, seq_len, embed_dim)
        # 输出out [128, 50] (batch, output_size)
        x = x.unsqueeze(1)  # [128, 1, 10, 100] (batch, channel_num, seq_len, embed_dim)
        x = [conv(x) for conv in self.convs]
        x = torch.cat(x, dim=1)  # [128, 300, 1, 1] 各通道的数据拼接在一起
        x = x.view(x.size(0), -1)  # [128, 300] 展平
        x = self.dropout(x)
        out = self.fc(x)
        return out


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.extractor = TextCNN_MODULE()
        self.fc1 = nn.Linear(2 * cnf.output_size, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, q1, q2):
        y1 = self.extractor(q1)  # (batch, output_size)
        y2 = self.extractor(q2)  # (batch, output_size)
        y1_y2 = torch.cat([y1, y2], dim=1)  # (batch, 2*output_size)
        out = self.fc2(F.relu(self.fc1(y1_y2)))  # (batch, 2)
        return out


class Similarity(nn.Module):
    def __init__(self, mode='cos'):
        super(Similarity, self).__init__()
        self.mode = mode

    def forward(self, y1, y2):
        # 输入y1 (batch, output_size)
        # 输入y2 (batch, output_size)
        # 输出sim (batch)
        if self.mode == 'cos':
            y1_y2 = torch.sum(y1 * y2, dim=1)
            y1_norm = torch.sqrt(torch.sum(y1 * y1, dim=1))
            y2_norm = torch.sqrt(torch.sum(y2 * y2, dim=1))
            sim = y1_y2 / (y1_norm * y2_norm)
            return sim
