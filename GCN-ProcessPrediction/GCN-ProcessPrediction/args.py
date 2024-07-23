import argparse

parser = argparse.ArgumentParser(description='GCN')
parser.add_argument("-d", "--dataset", type=str, default="bpi13_incidents", help="dataset to use")
parser.add_argument("--num-nodes", type=int, default=7)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--fold", type=int, default=0, help="3 fold")
args = parser.parse_args()