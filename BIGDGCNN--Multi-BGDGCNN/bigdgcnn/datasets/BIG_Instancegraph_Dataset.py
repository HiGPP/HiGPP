import torch
from torch_geometric.data import InMemoryDataset, Data

import pm4py.util.xes_constants as xes
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking

import networkx as nx
from typing import Tuple
import numpy as np

from bigdgcnn.data_processing import make_training_data, discover_model_imf
from bigdgcnn.util import add_artificial_start_end_events


class BIG_Instancegraph_Dataset(InMemoryDataset):
    def __init__(self,
                 eventlog: EventLog,
                 logname: str,
                 root="./data",
                 transform=None,
                 process_model: Tuple[PetriNet, Marking, Marking] = None,
                 imf_noise_thresh: float = 0,
                 force_reprocess=False):

        if process_model is None:

            self.eventlog = add_artificial_start_end_events(eventlog)
        else:
            self.eventlog = eventlog

        self.activities_index = list(
            sorted(list(set(evt[xes.DEFAULT_NAME_KEY] for case in self.eventlog for evt in case))))
        self.logname = logname
        self.activities_index
        self.process_model = process_model
        self.imf_noise_thresh = imf_noise_thresh
        self.force_reprocess = force_reprocess
        self.savefile = f"data_{self.logname}.pt"

        super().__init__(root, transform)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):

        return [self.savefile] if not self.force_reprocess else ["force-reprocess"]

    def process(self):
        print("---------------------现在是普通BIGDGCNN-------------------")

        if self.process_model is None:
            self.process_model = discover_model_imf(self.eventlog, self.imf_noise_thresh)

        self.activities_index = list(
            sorted(list(set(evt[xes.DEFAULT_NAME_KEY] for case in self.eventlog for evt in case))))

        data = make_training_data(self.eventlog, self.process_model, xes.DEFAULT_NAME_KEY)

        data_list = [self._networkx_to_pytorch_graph(graph, label) for graph, label in data]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _make_feature_vector(self, node):

        idx, activity = node
        ret = np.zeros(len(self.activities_index) + 1)
        ret[0] = idx
        ret[self.activities_index.index(activity) + 1] = 1
        return ret

    def _one_hot_encode(self, activity):

        ret = np.zeros(len(self.activities_index))
        ret[self.activities_index.index(activity)] = 1
        return torch.tensor(ret)

    def _networkx_to_pytorch_graph(self, graph: nx.DiGraph, label: str) -> Data:

        edge_index = torch.tensor(np.array([
            [edge[0][0], edge[1][0]]
            for edge in graph.edges
        ]), dtype=torch.int64)

        x = torch.tensor(
            np.array([self._make_feature_vector(node) for node in sorted(graph.nodes, key=lambda v: v[0])]),
            dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index.t().contiguous(),
            y=self._one_hot_encode(label),
        )
        return data
