import torch.nn as nn
import pickle
from config import args
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support
import torch
import os
from sklearn.preprocessing import LabelBinarizer
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, average_precision_score


def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


def multiclass_pr_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return average_precision_score(y_test, y_pred, average=average)

class MAAP_Test:
    def __init__(self, f):
        self.device = torch.device('cuda:{}'.format(args.gpu))
        self.fold = f
        self.criterion = nn.CrossEntropyLoss()


    @staticmethod
    def Union(lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    def mapping(self, df_train, df_valid, df_test, col):
        list_word = self.Union(self.Union(df_train[col].unique(), df_valid[col].unique()),df_test[col].unique())
        mapping = dict(zip(set(list_word), range(1, len(list_word) + 1)))
        len_mapping = len(set(list_word))
        return mapping, len_mapping

    def test(self, eval_model, list_view_test, num_view, cat_view):
        eval_model.eval()
        torch_dataset_test = Data.TensorDataset(*list_view_test)
        loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        att_str = cat_view + num_view + ['y']
        result_path = "result/" + args.eventlog
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        outfile2 = open(result_path + "/" + args.eventlog + "_" + ".txt", 'a')
        loss = 0
        acc = 0
        Y_test_int = []
        preds_a = []
        for batch_num, data in enumerate(loader_test):
            with torch.no_grad():
                att = []
                for i in range(len(att_str)):
                    att.append(data[i].to(self.device))
                outputs, enc_self_attns = eval_model(att_str, att)
                loss = self.criterion(outputs, att[-1])
                preds_a.append(outputs)
                Y_test_int.append(att[-1])
                loss += loss.item()
                acc += (outputs.argmax(1) == att[-1]).sum().item()
        Y_test_int = torch.cat(Y_test_int, 0).to('cpu')

        preds_a = torch.cat(preds_a, 0).to('cpu')
        preds_a = np.argmax(preds_a, axis=1)
        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                                       pos_label=None)

        auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
        prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")

        print(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write('\nAUC: ' + str(auc_score_macro))
        outfile2.write('\nPRAUC: ' + str(prauc_score_macro))
        outfile2.write('\n')

        outfile2.flush()
        outfile2.close()
        return loss / len(torch_dataset_test), acc / len(torch_dataset_test)

    def Final_test(self):
        with open("data/" + args.eventlog + "/" + args.eventlog + '_num_cols.pickle', 'rb') as pickle_file:
            num_view = pickle.load(pickle_file)
        with open("data/" + args.eventlog + "/" + args.eventlog + '_cat_cols.pickle', 'rb') as pickle_file:
            cat_view = pickle.load(pickle_file)
        print('Starting model...')
        list_cat_view_test = []
        for col in cat_view:
            list_cat_view_test.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_test.npy").astype(int)))
        list_num_view_test = []
        for col in num_view:
            list_num_view_test.append(torch.from_numpy(
                np.load("data/" + args.eventlog + "/" + args.eventlog + "_" + col + "_" + str(self.fold) + "_test.npy",
                        allow_pickle=True)).to(torch.float32))

        list_view_test = list_cat_view_test + list_num_view_test

        y_test = np.load("data/" + args.eventlog + "/" + args.eventlog + "_y_" + str(self.fold) + "_test.npy")

        y_test = torch.from_numpy(y_test - 1).to(torch.long)

        list_view_test.append(y_test)
        model_path = 'model/' + str(args.eventlog) + '/' + args.eventlog + '_' + str(args.n_layers) + '_' + str(args.n_heads) + '_' + str(args.epochs) + '_' + str(
            self.fold) + '_model.pkl'
        device = torch.device('cuda:0')
        check_model = torch.load(model_path, map_location=device)
        test_loss, test_acc = self.test(check_model, list_view_test, num_view, cat_view)
        print('-' * 89)
        print(f'\tLast_Loss: {test_loss:.5f}(test)\t|\tLast_Acc: {test_acc * 100:.2f}%(test)')
        print('-' * 89)


if __name__ == "__main__":
    for f in range(3):
        new_MAAP = MAAP_Test(f)
        new_MAAP.Final_test()