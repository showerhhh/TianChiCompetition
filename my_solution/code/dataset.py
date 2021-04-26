import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import config as cnf


def get_sentence_m(model, sentence):
    matrix = np.zeros((cnf.max_seq_len, cnf.embed_dim))
    mask = np.zeros(cnf.max_seq_len)
    k = min(len(sentence), cnf.max_seq_len)
    for i in range(k):
        try:
            matrix[i] = model.wv.vectors[model.wv.vocab[sentence[i]].index]
            mask[i] = 1
        except:
            continue
    return {'matrix': matrix, 'mask': mask}


def get_sentence_m_v2(sentence):
    matrix = np.zeros((cnf.max_seq_len, cnf.one_hot_dim))
    k = min(len(sentence), cnf.max_seq_len)
    for i in range(k):
        try:
            matrix[i][int(sentence[i])] = 1
        except:
            continue
    return {'matrix': matrix}


class MyDataset(Dataset):
    def __init__(self, mode):
        super(MyDataset, self).__init__()
        self.mode = mode

        if cnf.model_name == 'BERT':
            self.tokenizer = BertTokenizer.from_pretrained(cnf.bert_path)
            self.tokenizer.padding_side = 'right'
        else:
            self.w2v_model = Word2Vec.load(cnf.w2v_path)

        if mode == 'train':
            self.df = pd.read_table(cnf.train_path, names=['q1', 'q2', 'label']).fillna("0")
        elif mode == 'evaluate':
            self.df = pd.read_table(cnf.evaluate_path, names=['q1', 'q2', 'label']).fillna("0")
        elif mode == 'test':
            self.df = pd.read_table(cnf.test_path, names=['q1', 'q2']).fillna("0")

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.loc[index, :]
        item = {}

        if cnf.model_name == 'BERT':
            q1_tokens = self.tokenizer.tokenize(row['q1'])  # 检查q1具体的分词情况
            q2_tokens = self.tokenizer.tokenize(row['q2'])  # 检查q2具体的分词情况
            q1_tokens = self.tokenizer(row['q1'], max_length=cnf.max_seq_len, padding='max_length', truncation=True,
                                       return_tensors='pt')
            q2_tokens = self.tokenizer(row['q2'], max_length=cnf.max_seq_len, padding='max_length', truncation=True,
                                       return_tensors='pt')
            q1_tokens = {k: v.squeeze(0) for k, v in q1_tokens.items()}  # 将第0维去掉
            q2_tokens = {k: v.squeeze(0) for k, v in q2_tokens.items()}  # 将第0维去掉
            item['q1'] = q1_tokens
            item['q2'] = q2_tokens
        else:
            q1_sentence = row['q1'].split(' ')
            q2_sentence = row['q2'].split(' ')
            item['q1'] = get_sentence_m(self.w2v_model, q1_sentence)['matrix']
            item['q2'] = get_sentence_m(self.w2v_model, q2_sentence)['matrix']

        if self.mode in ['train', 'evaluate']:
            item['label'] = row['label']
        return item


if __name__ == '__main__':
    dataset = MyDataset(mode='train')
    dataloader = DataLoader(dataset, batch_size=cnf.batch_size, shuffle=True)

    for index, data in enumerate(dataloader):
        print(data['q1'].shape)
        print(data['q2'].shape)
        print(data['label'])
