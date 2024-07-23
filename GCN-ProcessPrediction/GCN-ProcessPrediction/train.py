import pandas as pd
import time
from datetime import datetime
import numpy as np
from args import args
from sklearn.metrics import accuracy_score

from scipy.linalg import fractional_matrix_power
from sklearn.metrics import classification_report
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm
from pm4py.visualization.dfg import visualizer as dfg_vis_fact
from torch._utils import _accumulate
from torch import randperm, default_generator
import math


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths, generator=default_generator):
    r"""
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    if sum(lengths) != len(dataset):
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = randperm(sum(lengths), generator=generator).tolist()
    return [Subset(dataset, indices[offset - length: offset]) for offset, length in zip(_accumulate(lengths), lengths)]


def generate_features(df, total_activities, num_features):
    lastcase = ''
    firstLine = True
    numlines = 0
    casestarttime = None
    lasteventtime = None
    features = []

    for i, row in df.iterrows():
        t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
        if row[0] != lastcase:
            casestarttime = t
            lasteventtime = t
            lastcase = row[0]
            numlines += 1
        timesincelastevent = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(lasteventtime))
        timesincecasestart = datetime.fromtimestamp(time.mktime(t)) - datetime.fromtimestamp(time.mktime(casestarttime))
        midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = datetime.fromtimestamp(time.mktime(t)) - midnight
        timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
        timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
        timediff3 = timesincemidnight.seconds
        timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday()
        lasteventtime = t
        firstLine = False
        feature_list = [timediff, timediff2, timediff3, timediff4]
        features.append(feature_list)

    df['Feature Vector'] = features

    firstLine = True
    NN_features = []

    for i, row in df.iterrows():
        if firstLine:
            features = np.zeros((total_activities, num_features))
            features[row[1] - 1] = row[3]
            firstLine = False
        else:
            if (row[3][0] == 0):
                features = np.zeros((total_activities, num_features))
                features[row[1] - 1] = row[3]
            else:
                features = np.copy(prev_row_features)
                features[row[1] - 1] = row[3]
        prev_row_features = features
        NN_features.append(features)

    return NN_features


def generate_labels(df, total_activities):
    next_activity = []
    next_timestamp = []

    for i, row in df.iterrows():
        if (i != 0):
            if (row[3][0] == 0):
                next_activity.append(total_activities)
            else:
                next_activity.append(row[1] - 1)
    next_activity.append(total_activities)
    for i, row in df.iterrows():
        if (i != 0):
            if (row[3][0] == 0):
                next_timestamp.append(0)
            else:
                next_timestamp.append(row[3][0])
    next_timestamp.append(0)

    return next_activity, next_timestamp


class EventLogData(Dataset):
    def __init__(self, input, output):
        self.X = input
        self.y = output
        self.y = self.y.to(torch.float32)
        self.y = self.y.reshape((len(self.y), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    def get_splits(self):
        train_idx = np.load("raw_dir/" + dataset_name + "/" + dataset_name + "_kfoldcv_" + str(
            f) + "/" + "train" + '_' + "index" + ".npy", allow_pickle=True)
        valid_idx = np.load("raw_dir/" + dataset_name + "/" + dataset_name + "_kfoldcv_" + str(
            f) + "/" + "valid" + '_' + "index" + ".npy", allow_pickle=True)
        test_idx = np.load("raw_dir/" + dataset_name + "/" + dataset_name + "_kfoldcv_" + str(
            f) + "/" + "test" + '_' + "index" + ".npy", allow_pickle=True)
        train = Subset(self, train_idx)
        valid = Subset(self, valid_idx)
        test = Subset(self, test_idx)
        return train, valid, test


def prepare_data_for_Predictor(NN_features, label):
    dataset = EventLogData(NN_features, label)
    train, valid, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=1, shuffle=True)
    valid_dl = DataLoader(valid, batch_size=1, shuffle=False)
    test_dl = DataLoader(test, batch_size=1, shuffle=False)
    return train_dl, valid_dl, test_dl


def generate_input_and_labels(path):
    df = pd.read_csv(path, sep=',', header=0, index_col=False)
    total_unique_activities = num_nodes
    NN_features = generate_features(df, total_unique_activities, num_features)
    next_activity, next_timestamp = generate_labels(df, total_unique_activities)
    NN_features = torch.Tensor(NN_features).to(torch.float32)
    next_activity = torch.Tensor(next_activity).to(torch.float32)
    next_timestamp = torch.Tensor(next_timestamp).to(torch.float32)

    train_dl, valid_dl, test_dl = prepare_data_for_Predictor(NN_features, next_activity)

    return train_dl, valid_dl, test_dl


def generate_process_graph(path):
    data = pd.read_csv(path, sep=',', header=0, index_col=False)
    num_nodes = data['ActivityID'].nunique()
    cols = ['case:concept:name', 'concept:name', 'time:timestamp']
    data.columns = cols
    data['time:timestamp'] = pd.to_datetime(data['time:timestamp'])
    data['concept:name'] = data['concept:name'].astype(str)
    log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
    dfg = dfg_algorithm.apply(log)
    if showProcessGraph:
        visualize_process_graph(dfg, log)
    max = 0
    min = 0
    adj = np.zeros((num_nodes, num_nodes))
    for k, v in dfg.items():
        for i in range(num_nodes):
            if (k[0] == str(i + 1)):
                for j in range(num_nodes):
                    if (k[1] == str(j + 1)):
                        adj[i][j] = v
                        if (v > max): max = v
                        if (v < min): min = v

    if binary_adjacency:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if (adj[i][j] != 0):
                    adj[i][j] = 1

    D = np.array(np.sum(adj, axis=1))
    D = np.matrix(np.diag(D))

    adj = np.matrix(adj)

    if laplacian_matrix:
        adj = D - adj

    if np.isclose(np.linalg.det(D), 0):
        epsilon = 1e-10
        D = D + epsilon * np.eye(D.shape[0])
    adj = fractional_matrix_power(D, -0.5) * adj * fractional_matrix_power(D, -0.5)
    adj = torch.Tensor(adj).to(torch.float)

    return adj


def visualize_process_graph(dfg, log):
    dfg_gv = dfg_vis_fact.apply(dfg, log, parameters={dfg_vis_fact.Variants.FREQUENCY.value.Parameters.FORMAT: "jpeg"})
    dfg_vis_fact.view(dfg_gv)
    dfg_vis_fact.save(dfg_gv, "dfg.jpg")


class GCNConv(torch.nn.Module):
    def __init__(self, num_nodes, num_features, out_channels):
        super(GCNConv, self).__init__()

        self.in_channels = num_features
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(num_features, out_channels))
        self.bias = Parameter(torch.Tensor(num_nodes))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, adj):
        x = adj @ x @ self.weight
        x = torch.flatten(x)
        x = x + self.bias
        return x


class EventPredictor(torch.nn.Module):
    def __init__(self, num_nodes, num_features=4):
        super(EventPredictor, self).__init__()

        self.layer1 = GCNConv(num_nodes, num_features, out_channels=1)
        self.layer2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_nodes, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(256, num_nodes + 1),
        )

    def forward(self, x, adj):
        x = self.layer1(x, adj)
        x = self.layer2(x)

        return x


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


if __name__ == '__main__':

    dataset_name = args.dataset
    f = args.fold
    path = "raw_dir/" + dataset_name + "/" + dataset_name + "_kfoldcv_" + str(f) + "/" + dataset_name + "_all.csv"

    num_nodes = args.num_nodes

    num_features = 4
    showProcessGraph = False

    device = f'cuda:{args.gpu}'

    num_epochs = args.epochs
    seed_value = 42

    weighted_adjacency = False
    binary_adjacency = True
    laplacian_matrix = True
    variant = 'laplacianOnBinary'

    num_runs = 1
    lr_value = args.lr

    run = 0
    for run in range(num_runs):
        print("Run: {}, Learning Rate: {}".format(run + 1, lr_value))
        model = EventPredictor(num_nodes, num_features)
        train_dl, valid_dl, test_dl = generate_input_and_labels(path)
        adj = generate_process_graph(path)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_value)

        model = model.to(device)
        adj = adj.to(device)
        epochs_plt = []
        acc_plt = []
        loss_plt = []
        valid_loss_plt = []

        for epoch in range(num_epochs):

            model.train()
            num_train = 0
            training_loss = 0
            predictions, actuals = list(), list()

            for i, (inputs, targets) in enumerate(train_dl):
                inputs, targets = inputs.to(device), targets.to(device)
                yhat = model(inputs[0], adj)

                loss = criterion(yhat.reshape((1, -1)), targets[0].to(torch.long))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

                training_loss += loss.item()

                yhat = yhat.to('cpu')
                yhat = torch.argmax(yhat)
                actual = targets.to('cpu')
                actual = actual[0]
                predictions.append(yhat)
                actuals.append(actual)
                num_train += 1

            with torch.no_grad():
                model.eval()
                num_valid = 0
                validation_loss = 0
                for i, (inputs, targets) in enumerate(valid_dl):
                    inputs, targets = inputs.to(device), targets.to(device)
                    yhat_valid = model(inputs[0], adj)
                    loss_valid = criterion(yhat_valid.reshape((1, -1)), targets[0].to(torch.long))
                    validation_loss += loss_valid.item()
                    num_valid += 1

            acc = accuracy_score(actuals, predictions)
            avg_training_loss = training_loss / num_train
            avg_validation_loss = validation_loss / num_valid

            if (epoch == 0):
                best_loss = avg_validation_loss
                torch.save(model.state_dict(), dataset_name + "_" + str(f) + 'EventPredictor_parameters_gcn.pt')

            if (avg_validation_loss < best_loss):
                torch.save(model.state_dict(), dataset_name + "_" + str(f) + 'EventPredictor_parameters_gcn.pt')
                best_loss = avg_validation_loss

            print("Epoch: {}, Loss: {}, Accuracy: {}, Validation loss : {}".format(epoch, avg_training_loss, acc,
                                                                                   avg_validation_loss))
            epochs_plt.append(epoch + 1)
            acc_plt.append(acc)
            loss_plt.append(avg_training_loss)
            valid_loss_plt.append(avg_validation_loss)

        model.load_state_dict(torch.load(dataset_name + "_" + str(f) + 'EventPredictor_parameters_gcn.pt'))

        result_path = "result/" + dataset_name
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        outfile2 = open(result_path + "/" + dataset_name + "_" + ".txt", 'a')
        all_preds = []
        all_labels = []
        with torch.no_grad():
            model.eval()
            for i, (inputs, targets) in enumerate(test_dl):
                inputs, targets = inputs.to(device), targets.to(device)
                yhat_test = model(inputs[0], adj)
                yhat_test = yhat_test.to('cpu')
                yhat_test = torch.argmax(yhat_test)
                actual = targets.to('cpu')
                actual = actual[0]
                all_preds.append(int(yhat_test.item()))
                all_labels.append(int(actual.item()))

        auc_score_macro = multiclass_roc_auc_score(all_labels, all_preds, average="macro")
        prauc_score_macro = multiclass_pr_auc_score(all_labels, all_preds, average="macro")

        print(classification_report(all_labels, all_preds, digits=3))
        outfile2.write(classification_report(all_labels, all_preds, digits=3))
        outfile2.write('\nAUC: ' + str(auc_score_macro))
        outfile2.write('\nPRAUC: ' + str(prauc_score_macro))
        outfile2.write('\n')

        outfile2.flush()
        outfile2.close()
