import argparse

parser = argparse.ArgumentParser(description="Next Activity Prediction.")
parser.add_argument("--eventlog", required=True, type=str, help="eventlog name")
parser.add_argument("--batch_size", default=64, type=int, help="batch size")
parser.add_argument("--epochs", default=50, type=int, help="number of total epochs")
parser.add_argument("--gpu", default=0, type=int, help="gpu")
parser.add_argument("--iterations", default=5, type=int, help="iterations")
parser.add_argument("--d_model", default=128, type=int, help="embedded dimension")
parser.add_argument("--d_ff", default=2048, type=int, help="feedForward dimension")
parser.add_argument("--n_layers", default=2, type=int, help="number of Encoder of Decoder Layer")
parser.add_argument("--n_heads", default=2, type=int, help="number of heads in Multi-Head Attention")
parser.add_argument("--warmup_steps", default=10, type=int, help="number of steps in warmup")
args = parser.parse_args()
