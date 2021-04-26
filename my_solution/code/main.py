import os

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from torch import nn, optim
# from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as cnf
from dataset import MyDataset
from earlystop import EarlyStop
from model.BERT import BERT

# 运行设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device: {}'.format(device))
# 模型
model = BERT().to(device)
print('Using model: {}'.format(model.__class__.__name__))
# 优化器
optimizer = optim.AdamW(model.parameters(), lr=cnf.lr)
# scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, threshold=1e-2)
# 损失函数
criterion = nn.CrossEntropyLoss()
# 数据提取器
train_dataset = MyDataset(mode='train')
train_dataloader = DataLoader(train_dataset, batch_size=cnf.batch_size, shuffle=True)
evaluate_dataset = MyDataset(mode='evaluate')
evaluate_dataloader = DataLoader(evaluate_dataset, batch_size=cnf.batch_size, shuffle=True)
test_dataset = MyDataset(mode='test')
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)


def train():
    early_stop = EarlyStop()
    for epoch in range(cnf.num_epochs):
        model.train()
        print('epoch_index={}, lr={}'.format(epoch, optimizer.param_groups[0]['lr']))
        train_loss = list()
        for index, data in enumerate(tqdm(train_dataloader), 1):
            if cnf.model_name == 'BERT':
                q1 = {k: v.to(device) for k, v in data['q1'].items()}  # (batch, seq_len)
                q2 = {k: v.to(device) for k, v in data['q2'].items()}  # (batch, seq_len)
            else:
                q1 = data['q1'].to(torch.float32).to(device)  # (batch, seq_len, embed_dim)
                q2 = data['q2'].to(torch.float32).to(device)  # (batch, seq_len, embed_dim)
            label = data['label'].to(torch.long).to(device)  # (batch)，均为0或1

            output = model(q1, q2)  # (batch, num_class)
            loss = criterion(output, label)

            train_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if index % 100 == 0:
                print('    batch_index={}, loss={:.5f}'.format(index, np.mean(train_loss)))
                train_loss = list()
        print("------------------Epoch Finish------------------")

        res = evaluate()
        flag = early_stop.check_earlystop(res, model)
        if flag:
            return
        # scheduler.step(res['auc'])
    print("------------------Train Finish------------------")


def evaluate():
    model.eval()
    losses = []
    outputs = []
    labels = []
    with torch.no_grad():
        for index, data in enumerate(tqdm(evaluate_dataloader), 1):
            if cnf.model_name == 'BERT':
                q1 = {k: v.to(device) for k, v in data['q1'].items()}  # (batch, seq_len)
                q2 = {k: v.to(device) for k, v in data['q2'].items()}  # (batch, seq_len)
            else:
                q1 = data['q1'].to(torch.float32).to(device)  # (batch, seq_len, embed_dim)
                q2 = data['q2'].to(torch.float32).to(device)  # (batch, seq_len, embed_dim)
            label = data['label'].to(torch.long).to(device)  # (batch)，均为0或1

            output = model(q1, q2)  # (batch, num_class)
            loss = criterion(output, label)

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
    score = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

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
            if cnf.model_name == 'BERT':
                q1 = {k: v.to(device) for k, v in data['q1'].items()}  # (batch, seq_len)
                q2 = {k: v.to(device) for k, v in data['q2'].items()}  # (batch, seq_len)
            else:
                q1 = data['q1'].to(torch.float32).to(device)  # (batch, seq_len, embed_dim)
                q2 = data['q2'].to(torch.float32).to(device)  # (batch, seq_len, embed_dim)

            output = model(q1, q2)  # (batch, num_class)
            out = torch.softmax(output, dim=1)[:, 1]  # (batch)
            predict_df = predict_df.append(pd.DataFrame({'out': round(out.item(), 3)}, index=[0]), ignore_index=True)
    predict_df.to_csv(cnf.result_path, header=False, sep='\n', index=False)
    print("------------------Predict Finish------------------")


if __name__ == '__main__':
    train()
    predict(cnf.model_name, cnf.lr, cnf.embed_dim)
