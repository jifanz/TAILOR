import copy

import torch.nn.functional as F
import wandb
import torch
import numpy as np
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import MultiLabelMetric, MultiClassMetric


class PassiveTrainer:
    def __init__(self, model_class, n_class, n_epochs, loss_fn, metric, pred_fn, batch_size=200, multi_label_flag=True,
                 pretrained=True):
        self.batch_size = batch_size
        self.model_class = model_class
        self.n_class = n_class
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.metric = metric
        self.pred_fn = pred_fn
        self.weighted = False
        self.multi_label_flag = multi_label_flag
        self.pretrained = pretrained

    def train(self, train_dataset, test_dataset, finetune=(None, None, None), log=False, lr=1e-4):
        model, loss_fn, n_epoch = finetune
        if model is None:
            model = self.model_class(self.n_class, pretrained=self.pretrained).cuda()
        else:
            model = copy.deepcopy(model).cuda()
        if loss_fn is None:
            loss_fn = self.loss_fn
        if n_epoch is None:
            n_epoch = self.n_epochs
        model.train()
        # optimizer = SGD(model.parameters(), lr=.1, momentum=.9, weight_decay=1e-4)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-5)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=40)

        for epoch in range(n_epoch):
            print(epoch)
            preds = np.zeros((len(train_dataset), self.n_class), dtype=float)
            labels = np.zeros((len(train_dataset), self.n_class), dtype=float)
            losses = np.zeros(len(train_dataset), dtype=float)
            counter = 0
            for img, target, *other in tqdm(loader):
                img, target = img.float().cuda(), target.float().cuda()
                pred = model(img).squeeze(-1)
                loss = loss_fn(pred, target, *other)

                preds[counter: (counter + len(pred))] = self.pred_fn(pred.data).cpu().numpy()
                labels[counter: (counter + len(pred))] = target.data.cpu().numpy()
                losses[counter: (counter + len(pred))] = loss.data.cpu().numpy()
                counter += len(pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if log and (((epoch + 1) % (n_epoch // 5) == 0) or (epoch == n_epoch - 1)):
                test_preds, test_labels, test_losses = self.test(train_dataset, model)
                log_dict = self.metric.compute(epoch, test_preds, test_labels, test_losses)
                wandb.log(log_dict)
        return model

    def test(self, dataset, model, ret_emb=False):
        model.eval()
        loader = DataLoader(dataset, batch_size=int(self.batch_size * 1.5), shuffle=False, num_workers=40)
        preds = np.zeros((len(dataset), self.n_class), dtype=float)
        labels = np.zeros((len(dataset), self.n_class), dtype=float)
        losses = np.zeros(len(dataset), dtype=float)
        embs = []
        counter = 0
        for img, target in tqdm(loader):
            img, target = img.float().cuda(), target.float().cuda()
            with torch.no_grad():
                if ret_emb:
                    pred, emb = model(img, ret_features=True)
                    pred, emb = pred.squeeze(-1), emb
                    embs.append(emb.cpu().numpy())
                else:
                    pred = model(img).squeeze(-1)
                loss = self.loss_fn(pred, target)
            preds[counter: (counter + len(pred))] = self.pred_fn(pred).cpu().numpy()
            labels[counter: (counter + len(pred))] = target.cpu().numpy()
            losses[counter: (counter + len(pred))] = loss.cpu().numpy()
            counter += len(pred)
        assert counter == len(preds)
        model.train()
        if ret_emb:
            return preds, labels, losses, np.concatenate(embs, axis=0)
        else:
            return preds, labels, losses


def get_fns(multi_label_flag):
    if multi_label_flag:
        loss_fn = F.binary_cross_entropy_with_logits
        pred_fn = torch.sigmoid
        metric = MultiLabelMetric()
    else:
        loss_fn = F.cross_entropy
        pred_fn = lambda x: torch.softmax(x, dim=-1)
        metric = MultiClassMetric()
    return loss_fn, pred_fn, metric


if __name__ == "__main__":
    from hyperparam import *
    from dataset import get_dataset
    from model import get_model_class

    lr = 1e-4
    batch_size = 250
    n_epoch = 20
    run_name = "model=%s, lr=%f, batch_size=%d, n_epoch=%d" % (model_name, lr, batch_size, n_epoch)
    wandb.init = wandb.init(project="Model Training, %s" % data_name, entity=wandb_name, name=run_name,
                            config=vars(args))
    train_dataset, val_dataset, test_dataset, multi_label_flag, n_class = get_dataset(data_name, batch_size)
    # train_dataset, _ = torch.utils.data.random_split(train_dataset, lengths=(
    #     len(train_dataset) // 10, len(train_dataset) - len(train_dataset) // 10))
    model_class = get_model_class(model_name)
    loss_fn, pred_fn, metric = get_fns(multi_label_flag)
    trainer = PassiveTrainer(model_class, n_class, n_epoch, loss_fn, metric, pred_fn, batch_size=batch_size)
    trainer.train(train_dataset, test_dataset, log=True, lr=lr)
