import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from datetime import datetime
import pickle
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer

class ReadLog:
    def __init__(self, eventlog):
        self._eventlog = eventlog
        self._list_cat_cols = []
        self._list_num_cols = []

    @staticmethod
    def Union(lst1, lst2):
        final_list = list(set(lst1) | set(lst2))
        return final_list

    @staticmethod
    def to_sec(c):
        sec = 86400 * c.days + c.seconds + c.microseconds / 1000000
        return sec / 86400

    def time_format(self, time_stamp):

        try:
            date_format_str = '%Y/%m/%d %H:%M:%S.%f'
            conversion = datetime.strptime(time_stamp, date_format_str)
        except:
            date_format_str = '%Y/%m/%d %H:%M:%S'
            conversion = datetime.strptime(time_stamp, date_format_str)
        return conversion

    def get_time(self, sequence, max_trace, mean_trace):
        i = 0
        s = (max_trace)
        list_seq = []
        while i < len(sequence):
            list_temp = []
            seq = np.zeros(s)
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(self.to_sec(self.time_format(sequence.iat[i, 0][j])-self.time_format(sequence.iat[i, 0][0])))  # 计算轨迹的持续时间存入列表
                new_seq = np.append(seq, list_temp)
                cut = len(list_temp)
                new_seq = new_seq[cut:]
                list_seq.append(new_seq[-mean_trace:])
                j = j + 1
            i = i + 1

        list_seq = np.array(list_seq)
        return list_seq

    def get_seq_view(self, sequence, max_trace, mean_trace):
        i = 0
        s = (max_trace)
        list_seq = []
        while i < len(sequence):
            list_temp = []
            seq = np.zeros(s)
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                new_seq = np.append(seq, list_temp)
                cut = len(list_temp)
                new_seq = new_seq[cut:]
                list_seq.append(new_seq[-mean_trace:])
                j = j + 1
            i = i + 1
        return list_seq


    def get_sequence(self, sequence, max_trace, mean_trace, next):
        i = 0
        s = (max_trace)
        list_seq = []
        list_label = []
        while i < len(sequence):
            list_temp = []
            seq = np.zeros(s)
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                new_seq = np.append(seq, list_temp)
                cut = len(list_temp)
                new_seq = new_seq[cut:]
                cls_list = [next]
                list_seq.append(np.append(cls_list, new_seq[-mean_trace:]))
                list_label.append(sequence.iat[i, 0][j + 1])
                j = j + 1
            i = i + 1
        return list_seq, list_label
    # 顺序编码函数(从1开始)
    def mapping(self, df_train, df_valid, df_test, col):
        list_word = self.Union(self.Union(df_train[col].unique(), df_valid[col].unique()),df_test[col].unique())
        mapping = dict(zip(set(list_word), range(1, len(list_word) + 1)))
        len_mapping = len(set(list_word))
        return mapping, len_mapping
    # 将属性映射为视图
    def mapping_cat(self, col, df_train, df_valid, df_test, max_trace, mean_trace, fold):
            if col == 'case':
                view_train = None
                view_test = None
            else:
                mapping, len_mapping = self.mapping(df_train, df_valid, df_test, col)
                df_train[col] = [mapping[item] for item in df_train[col]]
                df_valid[col] = [mapping[item] for item in df_valid[col]]
                df_test[col] = [mapping[item] for item in df_test[col]]
                view_train = df_train.groupby('case', sort=False).agg({col: lambda x: list(x)})
                view_valid = df_valid.groupby('case', sort=False).agg({col: lambda x: list(x)})
                view_test = df_test.groupby('case', sort=False).agg({col: lambda x: list(x)})

                if col == 'activity':
                    with open("data/" + self._eventlog + "/" + self._eventlog + '_y_' + str(
                            fold) + "_ActivityMap.pickle", 'wb') as f:
                        pickle.dump(mapping, f)
                    view_train, label_train = self.get_sequence(view_train, max_trace, mean_trace, len_mapping+1)
                    view_valid, label_valid = self.get_sequence(view_valid, max_trace, mean_trace, len_mapping + 1)
                    view_test, label_test = self.get_sequence(view_test, max_trace, mean_trace, len_mapping+1)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_train.npy", view_train)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_' + col + '_' + str(fold) + "_valid.npy", view_valid)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_test.npy", view_test)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_info.npy", len(mapping))
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_y_' + str(fold) + "_train.npy", label_train)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_y_' + str(fold) + "_valid.npy", label_valid)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_y_' + str(fold) + "_test.npy", label_test)
                    self._list_cat_cols.append(col)
                else:
                    view_train = self.get_seq_view(view_train, max_trace, mean_trace)
                    view_valid = self.get_seq_view(view_valid, max_trace, mean_trace)
                    view_test = self.get_seq_view(view_test, max_trace, mean_trace)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_train.npy",view_train)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_' + col + '_' + str(fold) + "_valid.npy", view_valid)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_' + col + '_'+  str(fold) + "_test.npy", view_test)
                    np.save("data/" + self._eventlog + "/" + self._eventlog + '_' + col + '_' + str(fold) + "_info.npy", len(mapping))
                    self._list_cat_cols.append(col)
    def readView(self):
        for fold in tqdm(range(3)):
            self._list_cat_cols = []
            self._list_num_cols = []
            df_train = pd.read_csv(
                "data/" + self._eventlog + "/" + self._eventlog + "_kfoldcv_" + str(fold) + "_train.csv", sep=',', header=0, index_col=False)
            df_valid = pd.read_csv(
                "data/" + self._eventlog + "/" + self._eventlog + "_kfoldcv_" + str(fold) + "_valid.csv", sep=',',
                header=0, index_col=False)
            df_test = pd.read_csv(
                "data/" + self._eventlog + "/" + self._eventlog + "_kfoldcv_" + str(fold) + "_test.csv", sep=',', header=0, index_col=False)
            # full_df = df_train.append(df_valid).append(df_test)
            full_df = pd.concat([df_train, df_valid, df_test])
            if 'timestamp' in df_train.columns:
                df_data = [df_train, df_valid, df_test]
                for train_test in df_data:
                    stamp = []
                    for caseid, group_data in train_test.groupby('case', sort=False):
                        time = group_data.loc[:, 'timestamp'].tolist()
                        for i in range(len(time)):
                            stamp.append(self.to_sec(self.time_format(time[i]) - self.time_format(time[0])))
                    train_test.loc[:, 'duration'] = pd.Series(stamp)
                source = df_train['duration'].to_numpy().reshape(-1, 1)
                target0 = df_valid['duration'].to_numpy().reshape(-1, 1)
                target = df_test['duration'].to_numpy().reshape(-1, 1)
                uniform = KBinsDiscretizer(n_bins=len(full_df['activity'].unique()), encode='ordinal', strategy='kmeans')
                uniform.fit(source)
                df_train['timestamp'] = uniform.transform(source).astype(str)
                df_valid['timestamp'] = uniform.transform(target0).astype(str)
                df_test['timestamp'] = uniform.transform(target).astype(str)
                df_train.drop(columns='duration', inplace=True)
                df_valid.drop(columns='duration', inplace=True)
                df_test.drop(columns='duration', inplace=True)

            # 统计完整数据集中不同轨迹的长度
            cont_trace = full_df['case'].value_counts(dropna=False)

            # 完整的数据集中最长的轨迹的长度
            max_trace = max(cont_trace)
            # 完整的数据集中轨迹的平均长度
            mean_trace = int(round(np.mean(cont_trace)))
            # 将训练集和测试集中的所有属性映射为视图
            for col in df_train.columns.tolist():
                if is_numeric_dtype(df_train[col]):
                    if df_train[col].isnull().any():
                        value = df_train[col].mode()[0]
                        df_train[col].fillna(value, inplace=True)
                        df_valid[col].fillna(value, inplace=True)
                        df_test[col].fillna(value, inplace=True)
                self.mapping_cat(col, df_train, df_valid, df_test, max_trace, mean_trace, fold)
        # 保存完整的数据集中轨迹的平均长度
        with open("data/" + self._eventlog + "/" + self._eventlog + '_seq_length.pickle', 'wb') as handle:
            pickle.dump(mean_trace, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # 保存类别属性的名称列表
        with open("data/" + self._eventlog + "/" + self._eventlog + '_cat_cols.pickle', 'wb') as handle:
            pickle.dump(self._list_cat_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # 保存数值属性的名称列表
        with open("data/" + self._eventlog + "/" + self._eventlog + '_num_cols.pickle', 'wb') as handle:
            pickle.dump(self._list_num_cols, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    list_eventlog = [
    # 'bpi12_all_complete',
    # 'bpi12w_complete',
    #'bpi13_incidents',
    'bpi13_problems',
     # 'bpic2017_o',
    #'bpic2020',
    # 'receipt',
    # 'bpi12_work_all',
    # "helpdesk",
    # 'bpi13_closed_problems',
# "bpi2020Prepaid_Travel_Costs",
#     "BPI2020_D"
                     ]
    for eventlog in tqdm(list_eventlog):
        for f in range(0,3):
            df_train = pd.read_csv("data/three_fold_data/" + eventlog + "/" + eventlog + "_kfoldcv_" + str(f) + "_train.csv", sep=',',
                                   header=0, index_col=False)
            df_test = pd.read_csv("data/three_fold_data/" + eventlog + "/" + eventlog + "_kfoldcv_" + str(f) + "_test.csv", sep=',',
                                  header=0, index_col=False)
            np.random.seed(133)
            grouped = df_train.groupby('case')
            new_order = np.random.permutation(list(grouped.groups.keys()))
            new_groups = [grouped.get_group(key) for key in new_order]
            log_shuffled = pd.concat(new_groups)
            log_shuffled.index = range(len(log_shuffled))
            train, valid = train_test_split(log_shuffled, test_size=0.2, shuffle=False)
            train.index = range(len(train))
            valid.index = range(len(valid))
            if not os.path.exists("data/" + eventlog):
                os.mkdir("data/" + eventlog)
            train.to_csv("data/" + eventlog + "/" + eventlog + "_kfoldcv_" + str(f) + "_train.csv", index=False)
            valid.to_csv("data/" + eventlog + "/" + eventlog + "_kfoldcv_" + str(f) + "_valid.csv", index=False)
            df_test.to_csv("data/" + eventlog + "/" + eventlog + "_kfoldcv_" + str(f) + "_test.csv", index=False)
        Multi_view = ReadLog(eventlog)
        Multi_view.readView()
