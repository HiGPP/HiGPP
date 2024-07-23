import os
import argparse
import time
import numpy as np
import pandas as pd

from processtransformer import constants
from processtransformer.data.processor import LogsDataProcessor

parser = argparse.ArgumentParser(
    description="Process Transformer - Data Processing.")


parser.add_argument("--dataset", 
    type=str, 
    default="bpi13_closed_problems",
    help="dataset name")

parser.add_argument("--dir_path", 
    type=str, 
    default="./datasets", 
    help="path to store processed data")

parser.add_argument("--raw_log_file_train",
    type=str,
    default="./datasets/bpi13_closed_problems/bpi13_closed_problems_kfoldcv_2_train.csv",
    # required=True,
    help="path to raw csv log file")

parser.add_argument("--raw_log_file_test",
    type=str,
    default="./datasets/bpi13_closed_problems/bpi13_closed_problems_kfoldcv_2_test.csv",
    # required=True,
    help="path to raw csv log file")
parser.add_argument("--fold",
    # required=True,
    default= 2,
    type=int,
    help="fold")

parser.add_argument("--task", 
    type=constants.Task, 
    default=constants.Task.NEXT_ACTIVITY,
    help="task name")

parser.add_argument("--sort_temporally", 
    type=bool, 
    default=False, 
    help="sort cases by timestamp")

args = parser.parse_args()

if __name__ == "__main__": 

    start = time.time()
    data_processor = LogsDataProcessor(name=args.dataset, 
        filepath_train=args.raw_log_file_train,
        filepath_test=args.raw_log_file_test,
        columns=["case", "activity", "timestamp"],
        fold=args.fold,
        dir_path=args.dir_path, pool=1)
    data_processor.process_logs(task=args.task, sort_temporally= args.sort_temporally)
    end = time.time()
    print(f"Total processing time: {end - start}")

