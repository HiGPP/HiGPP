
import torch.nn as nn
import dgl
import torch
from dgl.nn.pytorch.conv import SAGEConv
import torch.nn.functional as F
from utils import *
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='relu')


    def forward(self, x):
        x = self.fc1(x)

        x = F.relu(x)
        x = self.fc2(x)

        return x

class AttributeFusion(nn.Module):
    def __init__(self, hidden_dim, vocab_sizes, feature_list):
        super(AttributeFusion, self).__init__()
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(voca_size + 1, hidden_dim, padding_idx=0) for voca_size in vocab_sizes
        ])
        self.lstm_layers = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True) for _ in vocab_sizes
        ])
        self.linear_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in vocab_sizes
        ])
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.transform = MLP((len(feature_list) - 1) * hidden_dim,
                             ((len(feature_list) - 1) * hidden_dim + hidden_dim) * 2 // 3,
                             hidden_dim)
        self.feature_list = feature_list
        self.set_parameters()


    def set_parameters(self):

        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')


        for layer in self.linear_layers:

            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')



    def forward(self, g):
        list_embedding_feature = []
        for feature, embedding, rnn, linear in zip(self.feature_list, self.embedding_layers, self.lstm_layers,
                                                   self.linear_layers):
            if feature != "activity":
                padd_feature = g.ndata[feature]
                h_feature = embedding(padd_feature)
                lengths = (padd_feature != 0).sum(dim=1).cpu()
                packed_input = pack_padded_sequence(h_feature, lengths=lengths, batch_first=True,
                                                    enforce_sorted=False)
                output_packed, hn = rnn(packed_input)
                output, _ = pad_packed_sequence(output_packed, batch_first=True)
                last_valid_outputs = output[torch.arange(output.size(0)), lengths - 1]
                last_valid_outputs = F.relu(linear(last_valid_outputs))
                list_embedding_feature.append(last_valid_outputs)
            else:
                activity_feature = embedding(g.ndata[feature])

        h = torch.cat(list_embedding_feature, dim=1)
        h = F.relu(self.transform(h))
        h = self.linear(h + activity_feature)
        return h

class GraphConv(nn.Module):
    def __init__(self, hidden_dim, num_layers, vocab_sizes, feature_list, max_repeat_result):
        super(GraphConv, self).__init__()
        self.max_repeat_result = max_repeat_result
        self.attribute_fusion = AttributeFusion(hidden_dim, vocab_sizes, feature_list)

        self.hidden_dim = hidden_dim

        self.conv_layers = nn.ModuleList([
            SAGEConv(hidden_dim, hidden_dim, "lstm")
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.2)
        self.feature_list = feature_list
        self.norm_layers = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])



    def forward(self, g):
        h = F.relu(self.attribute_fusion(g))


        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            h_residual = h
            h = conv_layer(g, h)
            h = F.relu(norm_layer(h + h_residual))


        with g.local_scope():

            g.ndata['h'] = h

            hg_list = []  # Store the hg for each graph
            for subgraph in dgl.unbatch(g):
                num_nodes = len(subgraph.nodes())
                hg_list.append(subgraph.ndata['h'][num_nodes - 1])
            hg = torch.stack(hg_list, dim=0)
            return hg

class SAGEVirtualNodeClassifier(nn.Module):
    def __init__(self, feature_list, dataset, hidden_dim, n_classes, num_layers, fold, max_repeat_result):
        super(SAGEVirtualNodeClassifier, self).__init__()
        raw_dir_new = "./raw_dir" + '/' + dataset + '/' + dataset + "_kfoldcv_" + str(fold)
        vocab_sizes = [np.load(raw_dir_new + "/" + feature + "_info.npy", allow_pickle=True) for feature in
                       feature_list]
        self.feature_list = feature_list
        self.n_classes = n_classes + 1
        self.graph_conv = GraphConv(hidden_dim, num_layers, vocab_sizes, feature_list, max_repeat_result)


        self.classify = nn.Linear(hidden_dim, self.n_classes)



    def forward(self, g):
        local_feature = self.graph_conv(g)
        return self.classify(local_feature)





