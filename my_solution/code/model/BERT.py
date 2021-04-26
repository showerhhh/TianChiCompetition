import torch
import torch.nn.functional as F
from torch import nn
from transformers import BertTokenizer, BertModel

import config as cnf


class BERT_MODULE(nn.Module):
    def __init__(self):
        super(BERT_MODULE, self).__init__()
        self.bert = BertModel.from_pretrained(cnf.bert_path)
        # self.model_config = BertConfig.from_pretrained(cnf.bert_path)
        # self.model_config.output_hidden_states = True  # 设置返回所有隐层输出
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(768, cnf.output_size)

    def forward(self, tokens):
        last_hidden = self.bert(**tokens).last_hidden_state
        pooler = self.bert(**tokens).pooler_output
        # last_hidden层的输出，shape为[batch, seq_len, hidden_size]
        # pooler层的输出，shape为[batch, hidden_size]
        out = self.fc(pooler)
        return out


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.extractor = BERT_MODULE()
        self.fc1 = nn.Linear(2 * cnf.output_size, 2)

    def forward(self, q1, q2):
        y1 = self.extractor(q1)  # (batch, output_size)
        y2 = self.extractor(q2)  # (batch, output_size)
        y1_y2 = torch.cat([y1, y2], dim=1)  # (batch, 2*output_size)
        out = self.fc1(y1_y2)  # (batch, 2)
        return out


def practice():
    bert_path = '../../user_data/bert-base-uncased'

    # 运行设备
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('Using device: {}!'.format(device))

    # 加载bert分词器
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    # 将input分词
    q1 = "Who was Jim Henson ?"
    q2 = "Jim Henson was a puppeteer ."
    tokenizer.padding_side = 'right'
    # tokens = tokenizer.tokenize(q2)  # 检查q2具体的分词情况
    tokens = tokenizer(q1, max_length=5, padding='max_length', truncation=True, return_tensors='pt')
    # tokens = tokenizer(q1, q2, max_length=9, padding='max_length', truncation=True, return_tensors='pt')
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # 加载bert模型，这个路径文件夹下有bert_config.json配置文件和model.bin模型权重文件
    model = BertModel.from_pretrained(bert_path).to(device)
    model.eval()
    with torch.no_grad():
        # encoded_output为列表，长度12，是每一层transformer的输出，大小[batch_size, sequence_length, hidden_size]
        # 若output_all_encoded_layers=False，则encoded_output为最后一层transformer的输出，大小同上
        # pooled_output最后一层transformer的输出结果的第一个单词[cls]的hidden states，其已经蕴含了整个input句子的信息。
        # 大小[batch_size, hidden_size]
        # encoded_output, pooled_output = model(**tokens)
        # for out in encoded_output:
        #     print(out.shape)
        # print(pooled_output.shape)

        output = model(**tokens).pooler_output
        print(output.shape)


if __name__ == '__main__':
    practice()
