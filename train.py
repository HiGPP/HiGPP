from args import args
import pickle
from utils import *
import torch
import torch.nn as nn
from Dataset import BPICDataset
from dgl.dataloading import GraphDataLoader
from model.model import SAGEVirtualNodeClassifier
import copy
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train(model, train_loader, criterion, optimizer, device, data_length):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for batched_graph, labels in train_loader:
        batched_graph = batched_graph.to(device)
        labels = labels.to(device)

        logits = model(batched_graph)
        logits = logits.to(device)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += (logits.argmax(1) == labels).sum().item()
    return total_loss / data_length, total_accuracy / data_length


def validate(model, val_loader, criterion, device, data_length):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for batched_graph, labels in val_loader:
            batched_graph = batched_graph.to(device)
            labels = labels.to(device)

            logits = model(batched_graph)
            logits = logits.to(device)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            total_accuracy += (logits.argmax(1) == labels).sum().item()
    return total_loss / data_length, total_accuracy / data_length



def train_val(args):
    print("start training...")
    print("++++++++++MODEL STATISTICS++++++++")
    print(f"dataset: {args.dataset}")
    print(f"fold: {args.fold}")
    print(f"hidden_dim: {args.hidden_dim}")
    print(f"num_layers: {args.num_layers}")
    print(f"num_epochs: {args.num_epochs}")
    print(f"lr: {args.lr}")
    print(f"batch_size: {args.batch_size}")
    print(f"gpu: {args.gpu}")
    # 数据导入
    dataset_train = BPICDataset(name=args.dataset, type="train", fold=args.fold)
    dataset_val = BPICDataset(name=args.dataset, type="val", fold=args.fold)
    dataset_test = BPICDataset(name=args.dataset, type="test", fold=args.fold)
    raw_dir_new = "./raw_dir" + '/' + args.dataset + '/' + args.dataset + "_kfoldcv_" + str(args.fold)
    features_path = raw_dir_new + "/" + 'feature' + '_' + "list" + ".npy"
    n_classes_path = raw_dir_new + "/" + 'activity' + '_' + "info" + ".npy"
    max_repeat_path = raw_dir_new + "/" + "max_repeat_result" + ".npy"
    device = f'cuda:{args.gpu}'
    # device = 'cpu'
    with open(features_path, 'rb') as file:
        feature_list = pickle.load(file)
    print(feature_list)

    n_classes = np.load(n_classes_path, allow_pickle=True)
    max_repeat_result = np.load(max_repeat_path, allow_pickle=True)



    model = SAGEVirtualNodeClassifier(feature_list, args.dataset, args.hidden_dim, n_classes, args.num_layers, args.fold, max_repeat_result)
    train_loader = GraphDataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    val_loader = GraphDataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    test_loader = GraphDataLoader(dataset_test, batch_size=args.batch_size, shuffle=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    best_val_acc = 0
    patience = 10
    no_improvement_count = 0
    best_epoch = 0
    best_model = None
    # 训练循环
    for epoch in range(args.num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device, len(dataset_train))
        print(
            f'Epoch [{epoch + 1}/{args.num_epochs}], Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')
        val_loss, val_accuracy = validate(model, val_loader, criterion, device, len(dataset_val))
        print(
            f'Epoch [{epoch + 1}/{args.num_epochs}], Validation Accuracy: {val_accuracy:.4f}')
        scheduler.step(val_loss)
        if val_accuracy >= best_val_acc:
            best_val_acc = val_accuracy
            best_epoch = epoch + 1
            no_improvement_count = 0
            best_model = copy.deepcopy(model)
        else:
            no_improvement_count += 1
            if no_improvement_count >= patience:
                print("Early stopping: Validation accuracy has not improved for {} epochs.".format(patience))
                break

    path = os.path.join("save_model", args.dataset)
    if not os.path.exists(path):
        os.makedirs(path)
    model_path = 'save_model/' + str(args.dataset) + '/' + str(args.hidden_dim) + '_' + str(args.num_layers) + '_' + str(
        args.lr) + '_' + str(args.fold) + '_model.pkl'
    torch.save(best_model, model_path)
    check_model = torch.load(model_path)
    _, val_accuracy = validate(check_model, val_loader, criterion, device, len(dataset_val))
    print('-' * 89)
    print(
        f'Best_Epoch [{best_epoch:d}/{args.num_epochs}], Validation Accuracy: {val_accuracy:.4f}')
    print('-' * 89)
    _, test_accuracy = validate(check_model, test_loader, criterion, device, len(dataset_test))
    print(
        f'Best_Epoch [{best_epoch:d}/{args.num_epochs}], Test Accuracy: {test_accuracy:.4f}')
    print('-' * 89)



    print('Training finished.')


if __name__ == '__main__':

    train_val(args)