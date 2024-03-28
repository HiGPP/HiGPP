
from args import args

from sklearn.metrics import classification_report, precision_recall_fscore_support
from utils import *

import torch

from Dataset import BPICDataset
from dgl.dataloading import GraphDataLoader
import os
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
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

def test(model, val_loader, data_length, dataset_name, device):
    model.eval()
    total_accuracy = 0
    result_path = "result/" + dataset_name
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    outfile2 = open(result_path + "/" + dataset_name + "_" + ".txt", 'a')
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batched_graph, labels in val_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)

            logits = model(batched_graph)
            logits.to(device)
            all_preds.append(logits)
            all_labels.append(labels)
            # total_accuracy += compute_accuracy(logits, labels)
            total_accuracy += (logits.argmax(1) == labels).sum().item()
        all_preds = torch.cat(all_preds, 0).to('cpu')
        all_preds = np.argmax(all_preds, axis=1)
        all_labels = torch.cat(all_labels, 0).to('cpu')
        precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        auc_score_macro = multiclass_roc_auc_score(all_labels, all_preds, average="macro")
        prauc_score_macro = multiclass_pr_auc_score(all_labels, all_preds, average="macro")
        accuracy = total_accuracy / data_length
        print(classification_report(all_labels, all_preds, digits=3))
        outfile2.write(classification_report(all_labels, all_preds, digits=3))
        outfile2.write('\nAUC: ' + str(auc_score_macro))
        outfile2.write('\nPRAUC: ' + str(prauc_score_macro))
        outfile2.write('\n')

        outfile2.flush()
        outfile2.close()
    return accuracy, precision, recall, fscore, auc_score_macro, prauc_score_macro

if __name__ == '__main__':
    device = f'cuda:{args.gpu}'
    raw_dir_new = "./raw_dir" + '/' + args.dataset + '/' + args.dataset + "_kfoldcv_" + str(args.fold)

    dataset_test = BPICDataset(name=args.dataset, type="test", fold=args.fold)
    test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)
    model_path = 'save_model/' + str(args.dataset) + '/' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' + str(
        args.lr) + '_' + str(args.fold) + '_model.pkl'
    check_model = torch.load(model_path)
    test_accuracy, test_precision, test_recall, test_fscore, AUC, PR = test(check_model, test_loader,
                                                                            len(dataset_test), args.dataset, device)
    print(
        f'Test Accuracy: {test_accuracy:.4f}, '
        f'Test Precision:{test_precision:.4f}, '
        f'Test Recall:{test_recall:.4f}, '
        f'Test Fscore:{test_fscore:.4f}, '
        f'AUC:{AUC:.4f}, PR:{PR:.4f}'
    )
    print('-' * 89)




