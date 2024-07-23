import torch.nn as nn
import pickle
from config import args
import numpy as np
from TRM import multi_view_transformer as MVT
import torch
import copy
import os
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

class MAAP:
    def __init__(self, f):
        self.device = torch.device('cuda:{}'.format(args.gpu))
        self.fold = f
        self.warmup_steps = 10
        self.model =MVT.Transformer(args.eventlog, args.d_model, args.n_heads, args.n_layers, args.d_ff, self.fold).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        self.scheduler_cosine = CosineAnnealingLR(self.optimizer, args.epochs - self.warmup_steps)
        self.scheduler_warmup = LambdaLR(self.optimizer, lr_lambda=lambda epoch: epoch / self.warmup_steps)

    @staticmethod
    def Union(lst1, lst2):
        final_list = lst1 + lst2
        return final_list

    def train(self, list_view_train, num_view, cat_view, epoch):
        self.model.train()
        torch_dataset_train = Data.TensorDataset(*list_view_train)
        loader_train = Data.DataLoader(
            dataset=torch_dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        att_str = cat_view + num_view + ['y']
        train_loss = 0
        train_acc = 0
        for batch_num, data in enumerate(loader_train):
            att = []
            for i in range(len(att_str)):
                att.append(data[i].to(self.device))
            self.optimizer.zero_grad()
            outputs, enc_self_attns = self.model(att_str, att)
            loss = self.criterion(outputs, att[-1])
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            train_acc += (outputs.argmax(1) == att[-1]).sum().item()

        print(f'\tLoss: {train_loss / len(torch_dataset_train):.5f}(train)\t|\tAcc: {train_acc / len(torch_dataset_train) * 100:.2f}%(train)')

    def eval(self, eval_model, list_view_valid, num_view, cat_view):
        eval_model.eval()
        torch_dataset_valid = Data.TensorDataset(*list_view_valid)
        loader_valid = Data.DataLoader(
            dataset=torch_dataset_valid,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        att_str = cat_view + num_view + ['y']
        loss = 0
        acc = 0
        for batch_num, data in enumerate(loader_valid):
            with torch.no_grad():
                att = []
                for i in range(len(att_str)):
                    att.append(data[i].to(self.device))
                outputs, enc_self_attns = eval_model(att_str, att)
                loss = self.criterion(outputs, att[-1])
                loss += loss.item()
                acc += (outputs.argmax(1) == att[-1]).sum().item()
        return loss / len(torch_dataset_valid), acc / len(torch_dataset_valid)

    def train_val(self):
        with open("data/" + args.eventlog + "/" + args.eventlog + '_num_cols.pickle', 'rb') as pickle_file:
            num_view = pickle.load(pickle_file)
        with open("data/" + args.eventlog + "/" + args.eventlog + '_cat_cols.pickle', 'rb') as pickle_file:
            cat_view = pickle.load(pickle_file)
        best_val_acc = 0
        best_epoch = 0
        best_model = None
        print('Starting model...')
        list_cat_view_train = []
        for col in cat_view:
            list_cat_view_train.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_train.npy").astype(int)))

        list_cat_view_valid = []
        for col in cat_view:
            list_cat_view_valid.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_valid.npy").astype(int)))

        list_cat_view_test = []
        for col in cat_view:
            list_cat_view_test.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(
                    self.fold) + "_test.npy").astype(int)))

        list_num_view_train = []
        for col in num_view:
            list_num_view_train.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_train.npy",
                        allow_pickle=True)).to(torch.float32))

        list_num_view_valid = []
        for col in num_view:
            list_num_view_valid.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(
                    self.fold) + "_valid.npy",
                        allow_pickle=True)).to(torch.float32))

        list_num_view_test = []
        for col in num_view:
            list_num_view_test.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(
                    self.fold) + "_test.npy",
                        allow_pickle=True)).to(torch.float32))

        list_view_train = self.Union(list_cat_view_train, list_num_view_train)
        list_view_valid = self.Union(list_cat_view_valid, list_num_view_valid)
        list_view_test = self.Union(list_cat_view_test, list_num_view_test)

        y_train = np.load("data/" + args.eventlog + "/" + args.eventlog + "_y_" + str(self.fold) + "_train.npy")
        y_valid = np.load("data/" + args.eventlog + "/" + args.eventlog + "_y_" + str(self.fold) + "_valid.npy")
        y_test = np.load("data/" + args.eventlog + "/" + args.eventlog + "_y_" + str(self.fold) + "_test.npy")

        y_train = torch.from_numpy(y_train - 1).to(torch.long)
        y_valid = torch.from_numpy(y_valid - 1).to(torch.long)
        y_test = torch.from_numpy(y_test - 1).to(torch.long)

        list_view_train.append(y_train)
        list_view_valid.append(y_valid)
        list_view_test.append(y_test)

        for epoch in range(args.epochs):
            self.train(list_view_train, num_view, cat_view, epoch)
            valid_loss, valid_acc = self.eval(self.model, list_view_valid, num_view, cat_view)
            print('-' * 89)
            print(f'\tEpoch: {epoch:d}\t|\tLoss: {valid_loss:.5f}(valid)\t|\tAcc: {valid_acc * 100:.2f}%(valid)')
            print('-' * 89)

            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                best_epoch = epoch + 1
                best_model = copy.deepcopy(self.model)

            if epoch < self.warmup_steps:
                self.scheduler_warmup.step()
            else:
                self.scheduler_cosine.step()

        path = os.path.join("model", args.eventlog)
        if not os.path.exists(path):
            os.makedirs(path)
        model_path = 'model/' + str(args.eventlog) + '/' + args.eventlog + '_' + str(args.n_layers) + '_' + str(args.n_heads) + '_' + str(args.epochs) + '_' + str(
            self.fold) + '_model.pkl'
        torch.save(best_model, model_path)
        check_model = torch.load(model_path)
        valid_loss, valid_acc = self.eval(check_model, list_view_valid, num_view, cat_view)
        print('-' * 89)
        print(f'\tBest_Epoch: {best_epoch:d}\t|\tBest_Loss: {valid_loss:.5f}(valid)\t|\tBest_Acc: {valid_acc * 100:.2f}%(valid)')
        print('-' * 89)
        test_loss, test_acc = self.eval(check_model, list_view_test, num_view, cat_view)
        print('-' * 89)
        print(f'\tBest_Loss: {test_loss:.5f}(test)\t|\tBest_Acc: {test_acc * 100:.2f}%(test)')
        print('-' * 89)

if __name__ == "__main__":
    for f in range(3):
        new_MAAP = MAAP(f)
        new_MAAP.train_val()