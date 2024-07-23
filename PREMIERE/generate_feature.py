import itertools
import argparse
import os
import pickle

import pandas as pd
from tqdm import tqdm

from PREMIERE import PREMIERE

if __name__ == "__main__":

    list_eventlog = [
        'bpi12_all_complete',
        'bpi12w_complete',
        'bpi13_incidents',
        'bpi13_problems',
        'bpic2017_o',
        'bpic2020',
        'helpdesk',
        'receipt',
        'bpi12_work_all',
        'bpi13_closed_problems',

    ]

    for eventlog in tqdm(list_eventlog):
        for fold in range(3):
            dataset = eventlog
            print(f"\n开始处理： {eventlog}")

            pm = PREMIERE(dataset, fold=fold)

            log = pm.import_log()

            max_trace, n_caseid, n_activity = pm.dataset_summary(log=log)

            dict_card = {}
            n_col = 0
            for col in list(log.columns):
                if col != 'case' and col != 'timestamp':
                    dict_card[col] = log[col].nunique()
                    n_col = n_col + log[col].nunique()

            listOfevents = log['activity'].unique()
            listOfeventsInt = list(range(1, dict_card['activity'] + 1))

            train, test = pm.generate_prefix_trace(log=log)

            dict_view_train = {}
            dict_view_test = {}

            for col in list(log.columns):
                if col != 'case':
                    if col == 'timestamp':
                        dict_view_train[col] = pm.get_time(
                            train.groupby('case', sort=False).agg({col: lambda x: list(x)}))
                        dict_view_test[col] = pm.get_time(
                            test.groupby('case', sort=False).agg({col: lambda x: list(x)}))
                    else:
                        dict_view_train[col] = pm.get_sequence(
                            train.groupby('case', sort=False).agg({col: lambda x: list(x)}))
                        dict_view_test[col] = pm.get_sequence(
                            test.groupby('case', sort=False).agg({col: lambda x: list(x)}))

            agg_time_train = pm.agg_time_feature(dict_view_train['timestamp'])
            agg_time_test = pm.agg_time_feature(dict_view_test['timestamp'])

            flow_act = [p for p in itertools.product(listOfeventsInt, repeat=2)]

            target_train = pm.get_label(train.groupby('case', sort=False).agg({'activity': lambda x: list(x)}))
            target_test = pm.get_label(test.groupby('case', sort=False).agg({'activity': lambda x: list(x)}))

            print("train: \n", target_train)
            print("test: \n", target_test)

            premiere_feature_train = pm.premiere_feature(dict_view_train, flow_act, agg_time_train, target_train,
                                                         dict_card)
            premiere_feature_test = pm.premiere_feature(dict_view_test, flow_act, agg_time_test, target_test, dict_card)

            pm.generate_feature(dataset, premiere_feature_train, 'train')
            pm.generate_feature(dataset, premiere_feature_test, 'test')


            path = "image/" + dataset + f'_{fold}' + "/" + dataset
            if not os.path.exists(path):
                os.makedirs(path)

            with open(path + "_train_y.pkl", 'wb') as handle:
                pickle.dump(target_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(path + "_test_y.pkl", 'wb') as handle:
                pickle.dump(target_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

            print("feature generation complete")
