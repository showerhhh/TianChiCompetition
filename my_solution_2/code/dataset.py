import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader

import config as cnf


def process_sentence(model, sentence):
    tmp = [0] * cnf.max_seq_len
    k = min(len(sentence), cnf.max_seq_len)
    for i in range(k):
        try:
            tmp[i] = model.wv.vocab[sentence[i]].index
        except:
            continue
    return tmp


class MyDataset(Dataset):
    def __init__(self, mode):
        super(MyDataset, self).__init__()
        self.mode = mode
        self.w2v = Word2Vec.load(cnf.w2v_path)

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

        q1_sentence = row['q1'].split(' ')
        q2_sentence = row['q2'].split(' ')
        item['p'] = torch.tensor(process_sentence(self.w2v, q1_sentence)).to(torch.long)
        item['q'] = torch.tensor(process_sentence(self.w2v, q2_sentence)).to(torch.long)
        item['pf'] = torch.tensor(np.random.random(cnf.max_seq_len)).to(torch.float32)
        item['qf'] = torch.tensor(np.random.random(cnf.max_seq_len)).to(torch.float32)

        if self.mode in ['train', 'evaluate']:
            item['label'] = torch.tensor(row['label']).to(torch.float32)
        return item


if __name__ == '__main__':
    dataset = MyDataset(mode='train')
    dataloader = DataLoader(dataset, batch_size=cnf.batch_size, shuffle=False)

    for index, data in enumerate(dataloader):
        print(data['p'])
        print(data['q'])
        print(data['label'])
