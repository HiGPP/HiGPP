import os
import copy
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import xes_constants as xes
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score, \
    average_precision_score
from sklearn.preprocessing import LabelBinarizer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, Linear

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.aggr import SortAggregation
from torch_geometric import seed_everything

from sklearn.model_selection import train_test_split as sklearn_train_test_split
import networkx as nx
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

from bigdgcnn.data_processing import discover_model_imf
from bigdgcnn.util import add_artificial_start_end_events
from bigdgcnn.datasets import BIG_Instancegraph_Dataset
from bigdgcnn.datasets import BIG_Instancegraph_Dataset_With_Attributes


class DGCNN(nn.Module):

    def __init__(self,
                 dataset: Dataset,
                 graph_conv_layer_sizes: List[int],
                 sort_pool_k: int,
                 sizes_1d_convolutions: List[int],
                 dense_layer_sizes: List[int],
                 dropout_rate: float,
                 learning_rate: float,
                 activities_index: List[str],
                 use_cuda_if_available: bool = True,
                 ):
        """A Deep Graph Convolutional Neural Network (DGCNN) for next-activity prediction. Based on the paper "Exploiting Instance Graphs and Graph Neural Networks for Next Activity Prediction" by Chiorrini et al.

        Args:
            dataset (Dataset): The dataset to construct the model for.
            graph_conv_layer_sizes (List[int]): The output size for each graph convolution layer.
            sort_pool_k (int): The number of nodes to select in the SortPooling layer.
            sizes_1d_convolutions (List[int]): The output size for each 1D convolution layer.
            dense_layer_sizes (List[int]): The output size for each dense layer.
            dropout_rate (float): The dropout rate to use in the Dropout layer after the 1D convolution.
            learning_rate (float): The learning rate to use for the optimizer.
            activities_index (List[str]): The list of activities in the log. The same as used for one-hot encoding. Used to determine the number of input features.
            use_cuda_if_available (bool, optional): Whether to use CUDA if available. Defaults to True. If CUDA is not available, CPU will be used either way.
        """

        super().__init__()

        if use_cuda_if_available and torch.cuda.is_available():
            self.device = torch.device("cuda:1")
            print("Using GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")

        self.graph_conv_layer_sizes = graph_conv_layer_sizes
        self.sort_pool_k = sort_pool_k
        self.sizes_1d_convolutions = sizes_1d_convolutions
        self.dense_layer_sizes = dense_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        self.KERNEL_SIZE = min(sort_pool_k, len(graph_conv_layer_sizes))

        self.num_features = dataset.num_node_features
        self.num_output_features = len(activities_index)

        self.conv1 = SAGEConv(self.num_features, graph_conv_layer_sizes[0])
        self.convs = torch.nn.ModuleList()
        for in_size, out_size in zip(graph_conv_layer_sizes, graph_conv_layer_sizes[1:]):
            self.convs.append(SAGEConv(in_size, out_size))

        self.conv1d = Conv1d(graph_conv_layer_sizes[-1], sizes_1d_convolutions[0], self.KERNEL_SIZE)
        self.conv1ds = torch.nn.ModuleList()
        for in_size, out_size in zip(sizes_1d_convolutions, sizes_1d_convolutions[1:]):
            self.conv1ds.append(Conv1d(in_size, out_size, self.KERNEL_SIZE))

        self.linear = torch.nn.Linear(dense_layer_sizes[0] * (self.sort_pool_k - self.KERNEL_SIZE + 1),
                                      dense_layer_sizes[0])
        self.linears = torch.nn.ModuleList()
        for in_size, out_size in zip(dense_layer_sizes, dense_layer_sizes[1:]):
            self.linears.append(Linear(in_size, out_size))
        self.linear_output = Linear(dense_layer_sizes[-1], self.num_output_features)

    """加点初始化设置"""

    def reset_parameters(self):

        nn.init.xavier_uniform_(self.conv1.weight)
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight)

        nn.init.kaiming_uniform_(self.conv1d.weight, nonlinearity='relu')
        for conv1d in self.conv1ds:
            nn.init.kaiming_uniform_(conv1d.weight, nonlinearity='relu')

        nn.init.xavier_uniform_(self.linear.weight)
        for linear in self.linears:
            nn.init.xavier_uniform_(linear.weight)

        nn.init.xavier_uniform_(self.linear_output.weight)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        """数据中可能存在异常值，对异常值进行填充"""

        mean_value = torch.nanmean(x)
        x = torch.where(torch.isnan(x), mean_value, x)

        x = F.relu(self.conv1(x, edge_index))

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        module = SortAggregation(k=self.sort_pool_k)
        x = module(x, batch)

        x = x.view(len(x), self.sort_pool_k, -1).permute(0, 2,
                                                         1)

        x = F.relu(self.conv1d(x))
        for conv1d in self.conv1ds:
            x = F.relu(conv1d(x))

        x = x.view(len(x), -1)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = F.relu(self.linear(x))
        for linear in self.linears:
            x = F.relu(linear(x))
        x = F.relu(self.linear_output(x))
        return x

    def __repr__(self):
        params = {
            "Graph Conv. Layer Sizes": self.graph_conv_layer_sizes,
            "Sort Pool K": self.sort_pool_k,
            "1D Conv. Sizes": self.sizes_1d_convolutions,
            "Dense Layer Sizes": self.dense_layer_sizes,
            "Dropout Rate": self.dropout_rate,
            "Learning Rate": self.learning_rate,
            "Distinct Activities": self.num_output_features,
        }
        return self.__class__.__name__ + " Model: " + str(params)


class BIG_DGCNN():

    def __init__(self,
                 layer_sizes: List[int],
                 sort_pooling_k: int,
                 sizes_1d_convolutions: List[int] = [32],
                 dense_layer_sizes: List[int] = [32],
                 dropout_rate: float = 0.1,
                 learning_rate: float = 1e-3,
                 batch_size: int = 32,
                 epochs: int = 100,
                 seed: Optional[int] = None,
                 use_cuda_if_available: bool = True,
                 fold: int = None
                 ):
        """A Big-DGCNN model for next-activity prediction.

        Args:
            layer_sizes (List[int]): The output sizes of each graph convolution layer. Also determines the number of graph convolutions to perform. In the paper this count is varied in {2,3,5,7}, with each layer having size 32.
            sort_pooling_k (int): The number of nodes to select in the SortPooling layer. In the paper this is varied in {3, 5, 30}.
            sizes_1d_convolutions (List[int], optional): The sizes of the 1D convolutions. Defaults to [32], i.e., *one* 1D convolution with size 32, as used in the paper.
            dense_layer_sizes (List[int], optional): The sizes of the dense layers. Defaults to [32], i.e., *one* dense layer with size 32, as used in the paper.
            dropout_rate (float, optional): The dropout rate to use in the Dropout layer after the 1D convolution. Defaults to 0.1. In the paper this is varied in {0.1, 0.2, 0.5}
            learning_rate (float, optional): The learning rate to use for the optimizer. Defaults to 10^(-3). In the paper this is varied in {10^(-2), 10^(-3), 10^(-4)}
            batch_size (int, optional): The batch size to use for training. Defaults to 32. In the paper this is varied in {16, 32, 64, 128}
            epochs (int, optional): The number of epochs to train for. Defaults to 100, the value used in the paper.
            seed (int, optional): The seed to use for reproducibility. Defaults to None (No seed). Currently does not work as intended.
            use_cuda_if_available (bool, optional): Whether to use CUDA if available. Defaults to True.
        """
        if seed is not None:
            seed_everything(seed)
        self.seed = seed

        self.layer_sizes = layer_sizes
        self.sort_pooling_k = sort_pooling_k
        self.sizes_1d_convolutions = sizes_1d_convolutions
        self.dense_layer_sizes = dense_layer_sizes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.use_cuda_if_available = use_cuda_if_available
        self.fold = fold

    def train(self,
              log: EventLog,
              logname: str,
              process_model: Optional[Tuple[PetriNet, Marking, Marking]] = None,
              imf_noise_thresh: Optional[float] = None,
              train_test_split: float = 0.67,
              train_validation_split: float = 0.8,
              activityName_key: str = xes.DEFAULT_NAME_KEY,
              torch_load_path: Optional[str] = None,
              case_level_attributes: Optional[List[str]] = None,
              event_level_attributes: Optional[List[str]] = None,
              force_reprocess_dataset: bool = False
              ):
        """Train the Deep Graph Convolutional Neural Network (DGCNN) for next activity prediction on a log and a process model.

        Args:
            log (EventLog): The event log to use. Artificial start/end events will be added internally.
            logname (str): The name of the event log/dataset. Used for caching/loading the dataset.
            process_model (Tuple[PetriNet, Marking, Marking], optional): The Process Model (A Petri Next with initial and final marking). If not provided, one will be mined using the IMf Algorithm. Note that if you provide a model, it should contain the artificial start and end events already.
            imf_noise_thresh (float, optional): The noise threshold to be used for the Inductive Miner Infrequent algorithm if it is used (process_model is None). Defaults to None. If no process model is provided, and no imf_noise_thresh is provided, the lowest noise threshold (rounded to 10) generating a model with >=90% fitness will be used.
            activityName_key (str, optional): The key used for the activity label in the event log. Defaults to xes.DEFAULT_NAME_KEY (concept:name).
            torch_load_path (str, optional): The path to a saved model to load. If provided, any training will be skipped and the model will be loaded from the given path. Defaults to None.
            case_level_attributes (List[str], optional): The list of case-level attributes to use for training. Defaults to None, which means empty list.
            event_level_attributes (List[str], optional): The list of event-level attributes to use for training. Defaults to None, which means empty list.
        """
        case_level_attributes = case_level_attributes if case_level_attributes is not None else []
        event_level_attributes = event_level_attributes if event_level_attributes is not None else []

        log = add_artificial_start_end_events(log, activityName_key=activityName_key)
        if process_model is None:
            if imf_noise_thresh is not None:
                process_model = discover_model_imf(log, imf_noise_thresh)
            else:

                noise_thresh = 1.0
                condition = True
                while condition:
                    process_model = discover_model_imf(log, noise_thresh)
                    noise_thresh -= 0.1

                    fitness_results = replay_fitness.apply(
                        log,
                        *process_model,
                        variant=replay_fitness.Variants.TOKEN_BASED,
                        parameters={'show_progress_bar': False}
                    )
                    condition = fitness_results['percentage_of_fitting_traces'] < 0.9 and noise_thresh >= 0
                print(f"Discovered model using IMf with noise threshold {noise_thresh + 0.1}")
        self.process_model = process_model

        self.dataset = BIG_Instancegraph_Dataset(log,
                                                 logname + f"_{self.fold}_noisethresh_{int((noise_thresh + 0.1) * 100)}",
                                                 process_model=process_model)

        self.activities_index = self.dataset.activities_index

        self.train_split, self.test_split = sklearn_train_test_split(self.dataset, train_size=train_test_split,
                                                                     shuffle=False, random_state=self.seed)
        self.train_split, self.validation_split = sklearn_train_test_split(self.train_split,
                                                                           train_size=train_validation_split,
                                                                           shuffle=False, random_state=self.seed)

        self.train_data = DataLoader(self.train_split, batch_size=self.batch_size, shuffle=True)
        self.validation_data = DataLoader(self.validation_split, batch_size=self.batch_size, shuffle=True)
        self.test_data = DataLoader(self.test_split, batch_size=self.batch_size, shuffle=True)

        self.model = DGCNN(
            dataset=self.dataset,
            graph_conv_layer_sizes=self.layer_sizes,
            sort_pool_k=self.sort_pooling_k,
            sizes_1d_convolutions=self.sizes_1d_convolutions,
            dense_layer_sizes=self.dense_layer_sizes,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
            activities_index=self.activities_index,
            use_cuda_if_available=self.use_cuda_if_available
        )
        self.model = self.model.to(self.model.device)

        self._training_loop(logname)
        print(f"Training Completed")
        check_model = torch.load(self.model_path)

        self.metrics(self.test_data, logname, check_model)
        print(f"Accuracy on Test Set: {self.test_accuracy:.4f}%")

    def _training_loop(self, logname):
        self.model.train(True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        self.train_losses = []
        self.validation_losses = []

        self.train_accuracies = []
        self.validation_accuracies = []
        best_val_acc = 0
        patience = 20
        no_improvement_count = 0
        best_epoch = 0
        best_model = None
        epoch_losses = []
        for epoch in range(self.epochs):
            self.model.train(True)
            train_loss = 0
            for b in self.train_data:
                batch = b.to(self.model.device)
                optimizer.zero_grad(set_to_none=True)

                out = self.model(batch)
                label = batch.y.view(out.shape[0],
                                     -1)

                loss = criterion(out, label)

                loss.backward()

                optimizer.step()
                train_loss += loss.item()

            valid_loss = 0

            for b in self.validation_data:
                batch = b.to(self.model.device)
                out = self.model(batch)
                label = batch.y.view(out.shape[0], -1)
                loss = criterion(out, label)
                valid_loss += loss.item()
            this_epoch_losses = (
                train_loss / len(self.train_data), valid_loss / len(self.validation_data))
            epoch_losses.append(this_epoch_losses)

            valid_accuracy = self.evaluate(self.validation_data)
            print(
                f"Epoch {epoch + 1} completed. Train. Loss: {this_epoch_losses[0]}, Valid. Loss: {this_epoch_losses[1]}; Valid. Accuracy: {valid_accuracy * 100:.4f}%")

            self.train_losses.append(this_epoch_losses[0])
            self.validation_losses.append(this_epoch_losses[1])

            self.train_accuracies.append(
                self.evaluate(self.train_data))
            self.validation_accuracies.append(valid_accuracy)

            if valid_accuracy >= best_val_acc:
                best_val_acc = valid_accuracy
                no_improvement_count = 0
                best_model = copy.deepcopy(self.model)
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print("Early stopping: Validation accuracy has not improved for {} epochs.".format(patience))
                    break

            path = os.path.join("save_model/BIG", logname)
            if not os.path.exists(path):
                os.makedirs(path)
            self.model_path = 'save_model/BIG/' + logname + '/' + logname + '_model.pkl'
            torch.save(best_model, self.model_path)

    def evaluate(self, test_dataset: BIG_Instancegraph_Dataset) -> float:
        """Calculate the accuracy of the model on a test dataset.

        Args:
            test_dataset (BIG_Instancegraph_Dataset): The dataset to evaluate on.

        Returns:
            float: The computed accuracy
        """

        num_correct = 0
        total = 0
        self.model.eval()

        with torch.no_grad():
            for b in test_dataset:
                batch = b.to(self.model.device)

                out = self.model(batch)
                label = batch.y.view(out.shape)

                predictions = torch.argmax(out, dim=1)
                ground_truth = torch.argmax(label, dim=1).to(self.model.device)

                total += len(predictions)
                num_correct += torch.sum(predictions == ground_truth).item()
        return num_correct / total

    def multiclass_roc_auc_score(self, y_test, y_pred, average):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return roc_auc_score(y_test, y_pred, average=average)

    def multiclass_pr_auc_score(self, y_test, y_pred, average):
        lb = LabelBinarizer()
        lb.fit(y_test)
        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)
        return average_precision_score(y_test, y_pred, average=average)

    def metrics(self, test_dataset: BIG_Instancegraph_Dataset, logname, check_model):

        result_path = "results/BIG/" + logname

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        outfile2 = open(result_path + "/" + logname + "_" + ".txt", 'a')

        num_correct = 0
        total = 0
        check_model.eval()
        Y_labels = []
        Y_preds = []

        with torch.no_grad():
            for b in test_dataset:
                batch = b.to(check_model.device)

                out = check_model(batch)
                label = batch.y.view(out.shape)

                predictions = torch.argmax(out, dim=1)
                ground_truth = torch.argmax(label, dim=1).to(check_model.device)

                total += len(predictions)
                num_correct += torch.sum(predictions == ground_truth).item()

                Y_labels.append(ground_truth)
                Y_preds.append(predictions)

        Y_test_int = torch.cat(Y_labels, 0).to('cpu')

        preds_a = torch.cat(Y_preds, 0).to('cpu')

        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                                       pos_label=None)
        self.test_accuracy = (num_correct / total) * 100

        auc_score_macro = self.multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
        prauc_score_macro = self.multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")

        print(classification_report(Y_test_int, preds_a, digits=3))
        print(f"AUC:{auc_score_macro}")
        print(f"PRAUC:{prauc_score_macro}")

        outfile2.write(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write('\nAUC: ' + str(auc_score_macro))
        outfile2.write('\nPRAUC: ' + str(prauc_score_macro))
        outfile2.write('\n')

        outfile2.write('\n')

        outfile2.flush()

        outfile2.close()

    def evaluate_for_activity_set(self, test_dataset: BIG_Instancegraph_Dataset, activities: List[str],
                                  return_total: bool = False) -> float | Tuple[float, int]:
        """Calculate the accuracy of the model on a test dataset, but only for a given set of activities.
        I.e., consider only the datapoints which are labelled with one of the given activities.

        Args:
            test_dataset (BIG_Instancegraph_Dataset): The dataset to evaluate on.
            activities (List[str]): The list of activities to consider.
            return_total (bool, optional): Whether to return the total number of datapoints labelled with one of the considered activities. Defaults to False.

        Returns:
            float: The computed accuracy
        """
        num_correct = 0
        total = 0
        self.model.eval()

        with torch.no_grad():
            for b in test_dataset:
                batch = b.to(self.model.device)

                out = self.model(batch)
                label = batch.y.view(out.shape)

                predictions = torch.argmax(out, dim=1)
                ground_truth = torch.argmax(label, dim=1).to(self.model.device)

                considered = [
                    (prediction, ground_truth)
                    for prediction, ground_truth in zip(predictions, ground_truth)
                    if self.activities_index[ground_truth.item()] in activities
                ]
                total += len(considered)

                num_correct += len([x for x in considered if x[0] == x[1]])
        return num_correct / total if not return_total else (num_correct / total, total)

    def save(self, path: str):
        """Save the model using the torch.save function. Thus, the file will have the extension `.pt`.

        Args:
            path (str): The path to save the model to.
        """
        torch.save(self.model, path)

    def plot_training_history(self) -> plt.Figure:
        fig, ax = plt.subplots(2, 1, figsize=(8, 10))

        ax[1].plot(self.train_losses, label="Train")
        ax[1].plot(self.validation_losses, label="Validation")
        ax[1].set_ylabel("Loss", fontsize=14)

        ax[0].plot(self.train_accuracies, label="Train")
        ax[0].plot(self.validation_accuracies, label="Validation")
        ax[0].set_ylabel("Accuracy", fontsize=14)

        ax[1].set_xlabel("Epoch", fontsize=14)
        ax[0].legend(loc="upper right")

        return fig
