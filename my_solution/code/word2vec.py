import os
from collections import defaultdict

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

import config as cnf

df_train = pd.read_table("../user_data/oppo_breeno_round1_data/gaiic_track3_round1_train_20210228.tsv",
                         names=['q1', 'q2', 'label']).fillna("0")
df_test = pd.read_table('../user_data/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv',
                        names=['q1', 'q2']).fillna("0")

if not os.path.exists(cnf.train_path) or not os.path.exists(cnf.evaluate_path):
    train_shuffle = df_train.sample(frac=1.0)  # 全部打乱
    train_percent = 0.8
    cut_idx = int(round(train_percent * train_shuffle.shape[0]))
    train_sample, evaluate_sample = train_shuffle.iloc[:cut_idx, :], train_shuffle.iloc[cut_idx:, :]
    train_sample.to_csv(cnf.train_path, sep='\t', index=False, header=False)
    evaluate_sample.to_csv(cnf.evaluate_path, sep='\t', index=False, header=False)

if not os.path.exists(cnf.sentences_path):
    df_sentences = pd.DataFrame(columns=['sentence'])
    df_sentences = df_sentences.append(df_train['q1'])
    df_sentences = df_sentences.append(df_train['q2'])
    df_sentences = df_sentences.append(df_test['q1'])
    df_sentences = df_sentences.append(df_test['q2'])
    df_sentences.to_csv(cnf.sentences_path, header=False, sep='\n')

if not os.path.exists(cnf.w2v_path):
    sentences = LineSentence(cnf.sentences_path)
    # sentences = []
    # df_s = pd.read_table(cnf.sentences_path, sep='\n', header=None, names=['q'])
    # for index, row in df_s.iterrows():
    #     sentences.append(row['q'].split(' '))
    model = Word2Vec(sentences, size=cnf.embed_dim, window=5, min_count=5, workers=4)
    model.save(cnf.w2v_path)


def count():
    df = pd.read_table(cnf.sentences_path, sep='\n', header=None, names=['q'])
    length_count = defaultdict(int)
    for index, row in df.iterrows():
        length = len(row['q'].split(' '))
        length_count[length] += 1

    tmp = sorted(length_count.items(), key=lambda x: x[1], reverse=True)
    print(tmp)
