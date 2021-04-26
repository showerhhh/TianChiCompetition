import os

import numpy as np
import pandas as pd
import torch
from gensim.models import Word2Vec
from sklearn import metrics
from torch import nn, optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cnf
from dataset import MyDataset
from model.DRCN import DRCN

# 运行设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))
# word2vec矩阵
w2v = Word2Vec.load(cnf.w2v_path).wv.vectors
# 模型
model = DRCN(w2v).to(device)
# 优化器
optimizer = optim.AdamW(model.parameters(), lr=cnf.lr)
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, threshold=1e-2)
# 损失函数
criterion = nn.BCEWithLogitsLoss()
# 数据提取器
train_dataset = MyDataset(mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=cnf.batch_size, shuffle=True)
evaluate_dataset = MyDataset(mode='evaluate')
evaluate_dataloader = DataLoader(evaluate_dataset, batch_size=cnf.batch_size, shuffle=True)
test_dataset = MyDataset(mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def train():
    best_res = {
        "acc": 0.0,
        "prec": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "auc": 0.0,
        "ap": 0.0,
    }
    for epoch in range(cnf.num_epochs):
        model.train()
        print('epoch_index={}, lr={}'.format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = list()
        for index, data in enumerate(tqdm(train_dataloader), 1):
            p = data['p'].to(device)  # (batch, seq_len)
            q = data['q'].to(device)  # (batch, seq_len)
            pf = data['pf'].to(device)  # (batch, seq_len)
            qf = data['qf'].to(device)  # (batch, seq_len)
            label = data['label'].to(device)  # (batch)，均为0或1

            output, ae_loss = model(p, q, pf, qf)  # (batch, num_class)
            score = output[:, 1]  # (batch)
            loss = criterion(score, label) + ae_loss

            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % 100 == 0:
                print('    batch_index={}, loss={:.5f}'.format(index, np.mean(train_loss)))
                train_loss = list()
        print("------------------Epoch Finish------------------")

        res = evaluate()
        if res['auc'] >= best_res['auc']:
            best_res = res
            path = os.path.join(cnf.checkpoint_path,
                                '{}_lr_{}_embed_{}.pth'.format(model.__class__.__name__, cnf.lr, cnf.embed_dim))
            torch.save(model, path)
        # scheduler.step(res['auc'])
    print("------------------Train Finish------------------")


def evaluate():
    model.eval()
    losses = []
    outputs = []
    labels = []
    with torch.no_grad():
        for index, data in enumerate(tqdm(evaluate_dataloader), 1):
            p = data['p'].to(device)  # (batch, seq_len)
            q = data['q'].to(device)  # (batch, seq_len)
            pf = data['pf'].to(device)  # (batch, seq_len)
            qf = data['qf'].to(device)  # (batch, seq_len)
            label = data['label'].to(device)  # (batch)，均为0或1

            output, ae_loss = model(p, q, pf, qf)  # (batch, num_class)
            score = output[:, 1]  # (batch)
            loss = criterion(score, label) + ae_loss

            losses.append(loss.item())
            outputs.append(output)
            labels.append(label)

        evaluate_loss = np.mean(losses)
        outputs = torch.cat(outputs, dim=0)
        labels = torch.cat(labels, dim=0)
        res = metric(labels, outputs)

    print("loss: {}".format(evaluate_loss))
    print(res)
    print("------------------Evaluate Finish------------------")
    return res


def metric(y_true, outputs):
    # # 二分类
    # y_true = [0, 1, 1, 0, 1, 0]
    # y_pred = [0, 1, 0, 1, 1, 1]
    # score = [0.2, 0.7, 0.1, 0.5, 0.6, 0.9]
    # accuracy = metrics.accuracy_score(y_true, y_pred)  # 注意没有average参数
    # precision = metrics.precision_score(y_true, y_pred, average='binary')
    # recall = metrics.recall_score(y_true, y_pred, average='binary')
    # f1 = metrics.f1_score(y_true, y_pred, average='binary')
    # auc = metrics.roc_auc_score(y_true, score)
    # ap = metrics.average_precision_score(y_true, score)
    #
    # # 多分类
    # y_true = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4]
    # y_pred = [1, 1, 1, 0, 0, 2, 2, 3, 3, 3, 4, 3, 4, 3]
    # f1 = metrics.f1_score(y_true, y_pred, labels=[1, 2, 3, 4], average='micro')
    # f1 = metrics.f1_score(y_true, y_pred, labels=[1, 2, 3, 4], average='macro')
    # precision_class, recall_class, f1_class, _ = metrics.precision_recall_fscore_support(y_true=y_true,
    #                                                                                      y_pred=y_pred,
    #                                                                                      labels=[1, 2, 3, 4],
    #                                                                                      average=None)

    y_true = y_true.cpu().numpy()
    y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
    score = outputs[:, 1].cpu().numpy()

    acc = metrics.accuracy_score(y_true, y_pred)
    prec, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, labels=[1], average='macro')
    auc = metrics.roc_auc_score(y_true, score)
    ap = metrics.average_precision_score(y_true, score)

    res = {
        "acc": acc,
        "prec": prec,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "ap": ap
    }
    return res


def predict(model_name, lr, embed_dim):
    path = os.path.join(cnf.checkpoint_path, '{}_lr_{}_embed_{}.pth'.format(model_name, lr, embed_dim))
    model = torch.load(path).to(device)
    model.eval()
    predict_df = pd.DataFrame(columns=['out'])
    with torch.no_grad():
        for index, data in enumerate(tqdm(test_dataloader), 1):
            p = data['p'].to(device)  # (batch, seq_len)
            q = data['q'].to(device)  # (batch, seq_len)
            pf = data['pf'].to(device)  # (batch, seq_len)
            qf = data['qf'].to(device)  # (batch, seq_len)

            output, ae_loss = model(p, q, pf, qf)  # (batch, num_class)
            out = output[:, 1]  # (batch)
            out = torch.sigmoid(out)
            predict_df = predict_df.append(pd.DataFrame({'out': round(out.item(), 3)}, index=[0]), ignore_index=True)
    predict_df.to_csv(cnf.result_path, header=False, sep='\n', index=False)
    print("------------------Predict Finish------------------")


if __name__ == '__main__':
    train()
    predict('DRCN', cnf.lr, cnf.embed_dim)
