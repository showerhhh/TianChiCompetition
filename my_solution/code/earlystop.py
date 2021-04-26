import os

import torch

import config as cnf


class EarlyStop():
    def __init__(self, patience=4):
        super(EarlyStop, self).__init__()
        self.best_res = {
            "acc": 0.0,
            "prec": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auc": 0.0,
            "ap": 0.0,
        }
        self.patience = patience
        self.count = 0

    def check_earlystop(self, res, model):
        if res['auc'] >= self.best_res['auc']:
            self.best_res = res
            path = os.path.join(cnf.checkpoint_path,
                                '{}_lr_{}_embed_{}.pth'.format(cnf.model_name, cnf.lr, cnf.embed_dim))
            torch.save(model, path)
            self.count = 0
        else:
            self.count += 1

        if self.count >= self.patience:
            print("best_res: {}".format(self.best_res))
            print("------------------Early Stop Train------------------")
            return True
        else:
            return False
