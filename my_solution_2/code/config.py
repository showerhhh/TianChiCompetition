lr = 0.001
batch_size = 256
num_epochs = 20

max_seq_len = 12
num_embeds = 6925
num_classes = 2
embed_dim = 300
lstm_dim = 100
ae_dim = 200
fc_dim = 1000
embed_dp = 0.5
ae_dp = 0.2
lc_dp = 0.2

train_path = '../user_data/oppo_breeno_round1_data/train.tsv'
evaluate_path = '../user_data/oppo_breeno_round1_data/evaluate.tsv'
test_path = '../user_data/oppo_breeno_round1_data/gaiic_track3_round1_testA_20210228.tsv'

sentences_path = '../user_data/sentences.txt'
w2v_path = '../user_data/gensim_word2vec_' + str(embed_dim)

result_path = '../prediction_result/result.tsv'
checkpoint_path = '../user_data/checkpoint'
