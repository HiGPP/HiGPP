import argparse

parser = argparse.ArgumentParser(description='BPIC')
parser.add_argument("-d", "--dataset", type=str, default="bpi13_incidents", help="dataset to use")
parser.add_argument("--hidden-dim", type=int, default=128, help="dim of hidden")
parser.add_argument("--num-layers", type=int, default=2, help="number of layer")
parser.add_argument("--num-epochs", type=int, default=50, help="number of epoch")
parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
parser.add_argument("--batch-size", type=int, default=64, help="batch size")
parser.add_argument("--gpu", type=int, default=0, help="gpu")
parser.add_argument("--fold", type=int, default=0, help="3 fold")
args = parser.parse_args()