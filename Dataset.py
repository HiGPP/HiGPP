from dgl.data import DGLDataset
import pickle
import torch
import dgl
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import numpy as np
class BPICDataset(DGLDataset):
    def __init__(self,
                 name=None,
                 url=None,
                 raw_dir="./raw_dir",
                 save_dir=None,
                 force_reload=False,
                 verbose=False,
                 type=None,
                 fold=None):
        self.event_name = name
        self.type = type
        self.fold = fold
        super(BPICDataset, self).__init__(name=name,
                                        url=url,
                                        raw_dir=raw_dir,
                                        save_dir=save_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)


    def download(self):
        pass

    def process(self):
        raw_dir_new = self.raw_dir + '/' + self.event_name + '/' + self.event_name + "_kfoldcv_" + str(self.fold)
        self.graphs, self.label = self._load_graph(raw_dir_new)

    def _load_graph(self, path):
        type = self.type
        source_path = os.path.join(path, f"{type}_new_source.npy")
        destination_path = os.path.join(path, f"{type}_new_destination.npy")
        label_path = os.path.join(path, f"{type}_label.npy")
        max_repeat_path = os.path.join(path, f"max_repeat_result.npy")
        features_path = os.path.join(path, "feature_list.npy")

        source_lists = np.load(source_path, allow_pickle=True)
        destination_lists = np.load(destination_path, allow_pickle=True)
        label_lists = np.load(label_path, allow_pickle=True)
        labels = torch.tensor(label_lists, dtype=torch.int64)

        feature_lists = np.load(features_path, allow_pickle=True)

        max_repeat_result = np.load(max_repeat_path, allow_pickle=True)

        node_features = {}
        for feature_name in feature_lists:
            feature_path = os.path.join(path, f"{type}_{feature_name}.npy")
            att_lists = np.load(feature_path, allow_pickle=True)
            node_features[feature_name] = att_lists

        graphs = []

        for i in tqdm(range(labels.shape[0])):
            edges = (torch.tensor(source_lists[i]), torch.tensor(destination_lists[i]))
            g = dgl.graph(edges, idtype=torch.int32)
            for feature_name, feature_data in node_features.items():
                if feature_name != "activity":
                    padded_sequences = pad_sequence([torch.tensor(seq + [0] * (max_repeat_result - len(seq))) for seq in feature_data[i]],
                                                    batch_first=True, padding_value=0)
                    g.ndata[feature_name] = padded_sequences.to(dtype=torch.int32)
                else:
                    g.ndata[feature_name] = torch.tensor(feature_data[i], dtype=torch.int32)
            graphs.append(g)

        return graphs, labels

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)

