# -*- coding: utf-8 -*-
"""Copy of GCN_TimePredictor_(Training).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dGjxvgWOx7LkyMw3aJTimuqvfY0xLWGn

# Importing necessary packages and functions
"""

# !pip install pm4py
# !pip install torch-scatter==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
# !pip install torch-sparse==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
# !pip install torch-cluster==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
# !pip install torch-spline-conv==latest+cu101 -f https://pytorch-geometric.com/whl/torch-1.5.0.html
# !pip install torch-geometric

import pandas as pd
import time
from datetime import datetime
import numpy as np
from numpy import vstack
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from scipy.linalg import fractional_matrix_power

import torch 
import torch.nn as nn
from torch.nn import Parameter
#from torch_geometric.nn.inits import glorot, zeros

from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_algorithm
from pm4py.objects.conversion.dfg import converter as dfg_conv
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.visualization.dfg import visualizer as dfg_vis_fact
from pm4py.visualization.petrinet import visualizer as pn_vis

# from torch_geometric.nn.inits import glorot, zeros 

#Unable to import above line, so manually copy-pasting the source code

import math

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

import bisect
import warnings
from torch._utils import _accumulate
from torch import randperm, default_generator

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
    return [Subset(dataset, indices[offset - length : offset]) for offset, length in zip(_accumulate(lengths), lengths)]

"""# Setting the parameters"""

# Helpdesk dataset

path = '../Data/helpdesk.csv'
# path = '/content/drive/My Drive/MSc Dissertation/Data/helpdesk.csv'
save_folder = 'Results/helpdesk'
# save_folder = '/content/Results/helpdesk'
dataset = 'helpdesk'
num_nodes = 9

# # BPI dataset

# path = 'Data/bpi_12_w.csv'
# save_folder = 'Results/bpi'
# dataset = 'bpi'
# num_nodes = 6 

# # BPI dataset (No Repeats)

# path = '/Data/bpi_12_w_no_repeat.csv'
# save_folder = '/Results/bpi_no_repeat'
# num_nodes = 6 

num_features = 4
showProcessGraph = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
num_epochs = 100
seed_value = 42
# lr_value = 1e-05
weighted_adjacency = False
binary_adjacency = True
laplacian_matrix = False
variant = 'binary' # Choose from ['weighted','binary','laplacianOnWeighted','laplacianOnBinary']
num_runs = 5

"""# Data Pre-processing"""

def generate_features (df,total_activities,num_features):
  lastcase = ''
  firstLine = True
  numlines = 0
  casestarttime = None
  lasteventtime = None
  features = []

  for i,row in df.iterrows():
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0]!=lastcase:
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        numlines+=1
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    timediff3 = timesincemidnight.seconds #this leaves only time even occured after midnight
    timediff4 = datetime.fromtimestamp(time.mktime(t)).weekday() #day of the week
    lasteventtime = t
    firstLine = False
    feature_list = [timediff,timediff2,timediff3,timediff4]
    features.append(feature_list)

  df['Feature Vector'] = features
  
  firstLine = True
  NN_features =[]

  for i,row in df.iterrows():
    if firstLine:
      features = np.zeros((total_activities,num_features))
      features[row[1] - 1] = row[3]
      firstLine = False
    else:
      if (row[3][0] == 0):
        features = np.zeros((total_activities,num_features))
        features[row[1] - 1] = row[3]
      else:
        features = np.copy(prev_row_features)
        features[row[1] - 1] = row[3]
    prev_row_features = features
    NN_features.append(features)  
  
  return NN_features

def generate_labels(df,total_activities):
  next_activity = []
  next_timestamp = []

  for i,row in df.iterrows():
    if (i != 0):
      if (row[3][0]==0):
        next_activity.append(total_activities)
      else:
        next_activity.append(row[1]-1)
  next_activity.append(total_activities)
  for i,row in df.iterrows():
    if (i != 0):
      if (row[3][0]==0):
        next_timestamp.append(0)
      else:
        next_timestamp.append(row[3][0])
  next_timestamp.append(0)

  return next_activity,next_timestamp

class EventLogData(Dataset):
  def __init__ (self, input, output):
    self.X = input
    self.y = output
    self.y = self.y.to(torch.float32)
    self.y = self.y.reshape((len(self.y),1))

  #get the number of rows in the dataset
  def __len__(self):
    return len(self.X)

  #get a row at a particular index in the dataset
  def __getitem__ (self,idx):
    return [self.X[idx],self.y[idx]]
  
  # get the indices for the train and test rows
  def get_splits(self, n_test = 0.33, n_valid = 0.2):
    train_idx,test_idx = train_test_split(list(range(len(self.X))),test_size = n_test, shuffle = False )
    train_idx, valid_idx = train_test_split(train_idx, test_size = n_valid, shuffle = True)
    train = Subset(self, train_idx)
    valid = Subset(self, valid_idx)
    test = Subset(self, test_idx)
    return train, valid, test

def prepare_data_for_Predictor(NN_features,label):
  dataset = EventLogData(NN_features,label)
  train, valid, test = dataset.get_splits()
  train_dl = DataLoader(train, batch_size=1, shuffle = True)
  valid_dl = DataLoader(valid, batch_size=1, shuffle = False)
  test_dl = DataLoader(test, batch_size = 1, shuffle = False)
  return train_dl, valid_dl, test_dl

def generate_input_and_labels (path):
  df = pd.read_csv(path)
  total_unique_activities = num_nodes
  NN_features = generate_features(df,total_unique_activities,num_features)
  next_activity, next_timestamp = generate_labels(df,total_unique_activities)
  NN_features = torch.Tensor(NN_features).to(torch.float32)
  next_activity = torch.Tensor(next_activity).to(torch.float32)
  next_timestamp = torch.Tensor(next_timestamp).to(torch.float32)
 
  train_dl, valid_dl, test_dl = prepare_data_for_Predictor(NN_features, next_timestamp)
  
  return train_dl,valid_dl,test_dl

"""# Getting Adjacency Matrix from Process Graph"""

def generate_process_graph (path):
  data = pd.read_csv(path)
  num_nodes = data['ActivityID'].nunique() # 9 for helpdesk.csv
  cols = ['case:concept:name','concept:name','time:timestamp']
  data.columns = cols 
  data['time:timestamp'] = pd.to_datetime(data['time:timestamp'])
  data['concept:name'] = data['concept:name'].astype(str)
  log = log_converter.apply(data, variant=log_converter.Variants.TO_EVENT_LOG)
  dfg = dfg_algorithm.apply(log)
  if showProcessGraph:
    visualize_process_graph(dfg,log)
  max = 0
  min = 0
  adj = np.zeros((num_nodes,num_nodes))
  for k,v in dfg.items():
    for i in range(num_nodes):
      if(k[0] == str(i+1)):
        for j in range(num_nodes):
          if (k[1] == str(j+1)):
            adj[i][j] = v
            if (v > max): max=v
            if (v< min): min=v

  # print("Raw weighted adjacency matrix: {}".format(adj))
  
  if binary_adjacency:
    for i in range(num_nodes):
      for j in range(num_nodes):
        if (adj[i][j]!=0):
          adj[i][j]=1
    # print("Binary adjacency matrix: {}".format(adj))
  
  D = np.array(np.sum(adj, axis=1))
  D = np.matrix(np.diag(D))
  # print("Degree matrix: {}".format(D))
  
  adj = np.matrix(adj)

  if laplacian_matrix:
    adj = D - adj # Laplacian Transform 
    # print("Laplacian matrix: {}".format(adj))

  # adj = (D**-1)*adj
  adj = fractional_matrix_power(D, -0.5)*adj*fractional_matrix_power(D, -0.5)
  adj = torch.Tensor(adj).to(torch.float)
  
  # print("Symmetrically normalised Adjacency matrix: {}".format(adj))
  
  return adj

def visualize_process_graph (dfg,log):
  dfg_gv = dfg_vis_fact.apply(dfg, log, parameters={dfg_vis_fact.Variants.FREQUENCY.value.Parameters.FORMAT: "jpeg"})
  dfg_vis_fact.view(dfg_gv)
  dfg_vis_fact.save(dfg_gv,"dfg.jpg")

"""# Building Model"""

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
        x = adj@x@self.weight
        x = torch.flatten(x)
        x = x + self.bias
        return x

class TimePredictor(torch.nn.Module):
    def __init__(self,num_nodes, num_features = 4):
        super(TimePredictor, self).__init__()

        self.layer1 = GCNConv(num_nodes , num_features, out_channels=1)
        self.layer2 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(num_nodes,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,1),
        )

    def forward(self, x, adj):
        x = self.layer1(x,adj)
        x = self.layer2(x)
        return x

"""# Training the model"""

lr_run = 0
for lr_run in range(1):
  if lr_run==0:
    lr_value = 1e-03
  elif lr_run==1:
    lr_value = 1e-04
  elif lr_run==2:
    lr_value = 1e-05
  run = 0
  for run in range(num_runs):
    print("Run: {}, Learning Rate: {}".format(run+1,lr_value))
    model = TimePredictor(num_nodes, num_features)  
    train_dl, valid_dl, test_dl = generate_input_and_labels(path)
    adj = generate_process_graph(path)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr=lr_value)

    # print("************* Timestamp Predictor ***************")
    # print("Train size: {}, Validation size:{}, Test size: {}".format(len(train_dl.dataset),len(valid_dl.dataset),len(test_dl.dataset)))
    # print(model)
    model = model.to(device)
    adj = adj.to(device)
    epochs_plt = []
    mae_plt = []
    valid_loss_plt = []

    for epoch in range(num_epochs):
        
        model.train()
        training_loss = 0
        predictions, actuals = list(),list()
        num_train = 0

        for i, (inputs,targets) in enumerate(train_dl):

          inputs,targets = inputs.to(device), targets.to(device)
          optimizer.zero_grad() # Clearing the gradients

          yhat = model(inputs[0],adj)
          loss = criterion(yhat.reshape((1,-1)),targets[0].to(torch.long).reshape((1,-1)))
          loss.backward()
          optimizer.step()

          training_loss+= loss.item()
          num_train+=1

        with torch.no_grad():
          model.eval()
          num_valid = 0
          validation_loss = 0
          for i,(inputs,targets) in enumerate(valid_dl):
            inputs,targets = inputs.to(device),targets.to(device)
            yhat_valid = model(inputs[0],adj)
            loss_valid = criterion(yhat_valid.reshape((1,-1)),targets[0].to(torch.long).reshape((1,-1)))
            validation_loss+= loss_valid.item()
            num_valid+= 1

        avg_training_loss = training_loss/num_train
        avg_training_loss = avg_training_loss/86400
        avg_validation_loss = validation_loss/num_valid
        avg_validation_loss = avg_validation_loss/86400

        if (epoch==0): 
          best_loss = avg_validation_loss
          torch.save(model.state_dict(),'{}/TimestampPredictor_parameters_{}_{}_{}_run{}.pt'.format(save_folder,dataset,variant,lr_value,run))
        
        if (avg_validation_loss < best_loss):
          torch.save(model.state_dict(),'{}/TimestampPredictor_parameters_{}_{}_{}_run{}.pt'.format(save_folder,dataset,variant,lr_value,run))
          best_loss = avg_validation_loss
          
        print("Epoch: {}, Training MAE : {}, Validation loss : {}".format(epoch,avg_training_loss,avg_validation_loss))
        epochs_plt.append(epoch+1)
        mae_plt.append(avg_training_loss)
        valid_loss_plt.append(avg_validation_loss)

    filepath = '{}/Loss_{}_{}_{}_run{}.txt'.format(save_folder,dataset,variant,lr_value,run)

    with open(filepath, 'w') as file:
        for item in zip(epochs_plt,mae_plt,valid_loss_plt):
            file.write("{}\n".format(item))
