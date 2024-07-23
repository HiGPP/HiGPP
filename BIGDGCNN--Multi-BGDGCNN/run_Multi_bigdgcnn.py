from argparse import ArgumentParser
from multiprocessing import set_start_method

import torch

from bigdgcnn.ml.model_multi import BIG_DGCNN

parser = ArgumentParser()

parser.add_argument("--num_models", "-n", type=int, default=1)
parser.add_argument("--num_cores", "-c", type=int, help="Number of cores to use for multiprocessing", default=-1)
args = parser.parse_args()

from typing import List

import pm4py
from bigdgcnn.util import print_log_statistics
from torch.multiprocessing import Pool, cpu_count, freeze_support, spawn
from tqdm.auto import tqdm
from statistics import mean, stdev
from timeit import default_timer
from datetime import timedelta
from pathlib import Path
from tabulate import tabulate
import pandas as pd
import pandas as pd


def train_model(args) -> BIG_DGCNN:
    print("---------------------Multi-BIGDGCNN-------------------")

    all_log, event_attr, logname, fold = args

    model = BIG_DGCNN(
        sort_pooling_k=7,
        layer_sizes=[32] * 3,
        batch_size=64,
        learning_rate=1e-3,
        dropout_rate=0.1,
        sizes_1d_convolutions=[64],
        dense_layer_sizes=[64],
        epochs=100,

        use_cuda_if_available=True,
        fold=fold

    )

    model.train(all_log,
                logname=logname,
                case_level_attributes=[],
                event_level_attributes=event_attr)

    return model


def compute_new_attr(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['prev_activity'] = df.groupby('case')['activity'].shift(1)
    df['logical_predecessor'] = df.groupby('case')['activity'].apply(
        lambda x: x.shift(1).mask(x.shift(1) == x)).reset_index(drop=True)

    df['prev_timestamp'] = df.groupby('case')['timestamp'].shift(1)
    df['logical_predecessor_timestamp'] = df.groupby('case')['timestamp'].apply(
        lambda x: x.shift(1).mask(x.shift(1) == x)).reset_index(drop=True)

    df['delta_tni'] = (df['timestamp'] - df['logical_predecessor_timestamp']).dt.total_seconds().fillna(0)

    df['case_start_time'] = df.groupby('case')['timestamp'].transform('min')
    df['tdni'] = (df['timestamp'] - df['case_start_time']).dt.total_seconds()

    def get_week_start(dt):
        return dt - pd.Timedelta(days=dt.weekday())

    df['week_start_time'] = df['timestamp'].apply(get_week_start)
    df['twni'] = (df['timestamp'] - df['week_start_time']).dt.total_seconds()

    delta_maxe = df['delta_tni'].max()
    delta_maxt = df['tdni'].max()
    delta_tw = 7 * 24 * 60 * 60

    df['delta_tni'] = df['delta_tni'] / delta_maxe
    df['tdni'] = df['tdni'] / delta_maxt
    df['twni'] = df['twni'] / delta_tw

    df = df.drop(columns=['prev_activity', 'logical_predecessor', 'prev_timestamp', 'logical_predecessor_timestamp',
                          'case_start_time', 'week_start_time'])

    return df


def main():
    for fold in range(0, 3):

        df_train = pd.read_csv("fold/" + f"{eventlog}/" + eventlog + "_kfoldcv_" + str(fold) + "_train.csv", sep=',')
        df_test = pd.read_csv("fold/" + f"{eventlog}/" + eventlog + "_kfoldcv_" + str(fold) + "_test.csv", sep=',')

        all_data = pd.concat([df_train, df_test], ignore_index=True)

        all_data = compute_new_attr(all_data)

        excluded_columns = ["case", "activity", "timestamp"]
        filtered_columns = [str(col) for col in all_data.columns if col not in excluded_columns]

        event_attr = list(filtered_columns)

        all_data = pm4py.format_dataframe(all_data, case_id='case', activity_key='activity', timestamp_key='timestamp')
        all_log = pm4py.convert_to_event_log(all_data)

        print_log_statistics(all_log)
        train_model((all_log, event_attr, eventlog, fold))


if __name__ == '__main__':

    list_eventlog = [
        'helpdesk',
        'bpi13_problems',
        'bpi13_closed_problems',
        'bpi13_incidents',
        'bpic2017_o',
        'bpi12w_complete',
        'bpi12_all_complete',
        'bpi12_work_all',
        'receipt',
        'bpic2020',

    ]
    for event in tqdm(list_eventlog):

        eventlog = event
        main()
