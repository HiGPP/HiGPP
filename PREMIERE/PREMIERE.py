import os

import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from datetime import datetime
from Orange.data.pandas_compat import table_from_frame
import Orange
import csv
from itertools import tee, islice
import time


class PREMIERE:
    def __init__(self, eventlog, fold):
        self._eventlog = eventlog
        self.fold = fold

    def equifreq(self, log, n_bin):
        df = pd.DataFrame(log)
        df = table_from_frame(df)
        disc = Orange.preprocess.Discretize()
        disc.method = Orange.preprocess.discretize.EqualFreq(n=n_bin)
        df = disc(df)
        df = list(df)
        df = list(map(str, df))
        return df

    def import_log(self):

        print(f"-----------------------现在是{self._eventlog},开始第{self.fold}折-----------------------")

        df_train = pd.read_csv(
            "fold/" + f"{self._eventlog}/" + self._eventlog + "_kfoldcv_" + str(self.fold) + "_train.csv", sep=',')
        df_test = pd.read_csv(
            "fold/" + f"{self._eventlog}/" + self._eventlog + "_kfoldcv_" + str(self.fold) + "_test.csv", sep=',')

        log = pd.concat([df_train, df_test], ignore_index=True)

        if self._eventlog == 'bpi12w_complete' or self._eventlog == 'bpi12_all_complete' or self._eventlog == 'bpi12_work_all':
            log['resource'] = 'Res' + log['resource'].astype(str)

        list_columns = list(log.columns)

        num_act = len(list(log['activity'].unique()))

        num_res = len(list(log['resource'].unique()))

        n_bin = int((num_act + num_res) / 2)

        for col in list_columns:

            if col == 'case' or col == 'timestamp':
                print('skip')

            else:

                if is_numeric_dtype(log[col]):
                    log[col] = self.equifreq(log[col], n_bin)

                unique = log[col].unique()

                dictOfWords = {i: unique[i] for i in range(0, len(unique))}

                dictOfWords = {v: k for k, v in dictOfWords.items()}

                for k in dictOfWords:
                    dictOfWords[k] += 1

                log[col] = [dictOfWords[item] for item in log[col]]

        return log


    def generate_prefix_trace(self, log):
        grouped = log.groupby("case")
        start_timestamps = grouped["timestamp"].min().reset_index()
        start_timestamps = start_timestamps.sort_values("timestamp", ascending=True, kind="mergesort")
        train_ids = list(start_timestamps["case"])[:int(0.66 * len(start_timestamps))]

        train = log[log["case"].isin(train_ids)].sort_values("timestamp", ascending=True, kind='mergesort')
        test = log[~log["case"].isin(train_ids)].sort_values("timestamp", ascending=True, kind='mergesort')

        return train, test

    def get_label(self, act):
        i = 0
        list_label = []
        while i < len(act):
            j = 0
            while j < (len(act.iat[i, 0]) - 1):
                list_label.append(act.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        return list_label

    @staticmethod
    def dataset_summary(log):
        print("Activity Distribution\n", log['activity'].value_counts())
        n_caseid = log['case'].nunique()
        n_activity = log['activity'].nunique()
        print("Number of CaseID", n_caseid)
        print("Number of Unique Activities", n_activity)
        print("Number of Activities", log['activity'].count())
        cont_trace = log['case'].value_counts(dropna=False)
        max_trace = max(cont_trace)
        print("Max length trace", max_trace)
        print("Mean length trace", np.mean(cont_trace))
        print("Min length trace", min(cont_trace))
        return max_trace, n_caseid, n_activity

    def output_list(self, masterList):
        output = []
        for item in masterList:
            if isinstance(item, list):
                for i in self.output_list(item):
                    output.append(i)
            else:
                output.append(item)
        return output

    def generate_feature(self, namedataset, premiere_feature, typeset):
        path = "feature_fold/" + namedataset + f'_{self.fold}' + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + namedataset + "_" + typeset + ".csv", "w", newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for feature in premiere_feature:
                writer.writerow(self.output_list(feature))

    @staticmethod
    def ngrams(iterable, n):
        iters = tee(iterable, n)
        for i, it in enumerate(iters):
            next(islice(it, i, i), None)
        return zip(*iters)

    def premiere_feature(self, dict_view, flow_act, agg_time_feature, target, dict_card):
        list_flow_feature = []

        list_cols = list(dict_view.keys())
        list_cols.remove('timestamp')

        for j in range(len(dict_view['activity'])):
            n_list = {}
            for v in list_cols:
                n_list[v] = dict.fromkeys(list(range(1, dict_card[v] + 1)), 0)
                for item in dict_view[v][j]:
                    if item in n_list[v]:
                        n_list[v][item] += 1

            list_view_feature = [list(v.values()) for v in n_list.values()]

            list_gram = list(self.ngrams(dict_view['activity'][j], 2))
            flow_feature = [list_gram.count(flow) for flow in flow_act]
            list_agg_count = np.concatenate(list_view_feature)
            list_flow_feature.append(flow_feature + list_agg_count.tolist() + agg_time_feature[j] + [target[j]])

        return list_flow_feature

    def get_time(self, sequence):
        i = 0
        list_seq = []
        datetimeFormat = '%Y/%m/%d %H:%M:%S'
        while i < len(sequence):
            list_temp = []
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                t = time.strptime(sequence.iat[i, 0][0 + j], datetimeFormat)
                list_temp.append(datetime.fromtimestamp(time.mktime(t)))
                list_seq.append(list_temp.copy())
                j = j + 1
            i = i + 1
        return list_seq

    def get_sequence(self, sequence):
        i = 0
        list_seq = []
        list_label = []
        while i < len(sequence):
            list_temp = []
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                list_seq.append(list_temp.copy())
                list_label.append(sequence.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        return list_seq

    def pairwise(self, iterable):
        "s -> (s0, s1), (s1, s2), (s2, s3), ..."
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def agg_time_feature(self, time_prefix_new):
        i = 0
        list_agg_time_feature = []
        while i < len(time_prefix_new):
            time_feature = []
            duration = time_prefix_new[i][-1] - time_prefix_new[i][0]

            time_feature.append((86400 * duration.days + duration.seconds + duration.microseconds / 1000000) / 86400)
            time_feature.append(len(time_prefix_new[i]))
            if len(time_prefix_new[i]) == 1:

                time_feature.append(0)
                time_feature.append(0)
                time_feature.append(0)
                time_feature.append(0)
            else:

                diff_cons = [y - x for x, y in self.pairwise(time_prefix_new[i])]
                diff_cons_sec = [((86400 * item.days + item.seconds + item.microseconds / 1000000) / 86400) for item in
                                 diff_cons]
                time_feature.append(np.mean(diff_cons_sec))
                time_feature.append(np.median(diff_cons_sec))
                time_feature.append(np.min(diff_cons_sec))
                time_feature.append(np.max(diff_cons_sec))
            list_agg_time_feature.append(time_feature)

            i = i + 1
        return list_agg_time_feature
