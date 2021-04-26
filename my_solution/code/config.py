model_name = 'BERT'  # 'TextCNN' 'LSTM'

lr = 5e-5
batch_size = 256
num_epochs = 80

max_seq_len = 12  # 一句话的长度
embed_dim = 512  # 词向量长度
one_hot_dim = 7000  # one-hot向量长度
output_size = 300  # 句子向量长度
dropout = 0.5

# LSTM
hidden_size = 64
num_layers = 2

train_path = '../user_data/oppo_breeno_round1_data/train.tsv'
evaluate_path = '../user_data/oppo_breeno_round1_data/evaluate.tsv'
test_path = '../user_data/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv'

sentences_path = '../user_data/sentences.txt'
w2v_path = '../user_data/gensim_word2vec_' + str(embed_dim)

result_path = '../prediction_result/result.tsv'
checkpoint_path = '../user_data/checkpoint'

bert_path = '../user_data/bert-base-uncased'
