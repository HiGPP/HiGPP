import numpy as np
import pandas as pd
import pickle
import os
from args import args
import warnings
from tqdm import tqdm
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from utils import DataProcessingTools
# from utils import to_duration, encode_map, get_prefix_sequence_label, get_prefix_sequence

warnings.filterwarnings(action='ignore')  # 忽略警告


class EventLogProcessor:
    def __init__(self, event_name):
        self.event_name = event_name
        self.data_dir = f"raw_dir/{self.event_name}"
        self.three_fold_dir = f"raw_dir/three_fold_data/{self.event_name}"
        self.feature_list = []

    def edges_process(self, event_log):
        group_process_list = []
        for group_id, group in event_log.groupby('case', sort=False):
            group = group.copy()
            group = pd.concat([group.iloc[0:1], group])
            group["activity:next"] = group["activity"]
            group["activity:next"] = group["activity:next"].shift(-1)
            group = group.iloc[:-1, :]
            group = group.loc[:, ["case", "activity", "activity:next"]]
            group_process_list.append(group)
        edges_raw = pd.concat(group_process_list)
        edges_raw.rename(columns={"activity": "source", "activity:next": "destination"}, inplace=True)
        return edges_raw

    def add_attribute_process(self, event_log):
        group_process_list = []
        for group_id, group in event_log.groupby('case', sort=False):
            group = group.copy()
            group['occurrences'] = group.groupby("activity", sort=False).cumcount() + 1
            group['repeat'] = group['occurrences'].map(lambda x: 2 if x > 1 else 1)
            group_process_list.append(group)
        add_attribute_raw = pd.concat(group_process_list)
        return add_attribute_raw

    def save_edges_feature(self, raw, type, fold):
        for col in raw.columns.tolist():
            if col == "case":
                continue
            sequence = raw.groupby("case", sort=False).agg({col: lambda x: list(x)})
            if col == "destination":
                list_seq, list_label, list_len = DataProcessingTools.get_prefix_sequence_label(sequence)
                self.save_to_file(list_seq, f"{type}_{col}", fold)
                self.save_to_file(list_label, f"{type}_label", fold)
                self.save_to_file(list_len, f"{type}_len", fold)
            elif col == "source":
                list_seq = DataProcessingTools.get_prefix_sequence(sequence)
                self.save_to_file(list_seq, f"{type}_{col}", fold)

    def save_node_feature(self, nodes_raw, type, fold):
        source_lists = self.load_from_file(f"{type}_source", fold)
        destination_lists = self.load_from_file(f"{type}_destination", fold)
        group_process_list = []
        index = 0
        new_source_lists = []
        new_destination_lists = []

        for group_id, group in nodes_raw.groupby('case', sort=False):
            i = 1
            while i < len(group):
                group_temp = group[:i]
                unique_values = group_temp['activity'].drop_duplicates(keep='last')
                df_activity = pd.DataFrame(unique_values)
                group_temp = group_temp.groupby('activity', sort=False).agg(
                    lambda x: x.tolist()).reset_index()
                group_temp = pd.merge(df_activity, group_temp, on='activity', how='left')
                group_temp['id'] = index
                group_process_list.append(group_temp)
                encoding_map = {value: code for code, value in enumerate(unique_values)}
                source_list = [encoding_map[value] for value in source_lists[index]]
                new_source_lists.append(source_list)
                destination_list = [encoding_map[value] for value in destination_lists[index]]
                new_destination_lists.append(destination_list)
                i = i + 1
                index = index + 1

        self.save_to_file(new_source_lists, f"{type}_new_source", fold)
        self.save_to_file(new_destination_lists, f"{type}_new_destination", fold)
        nodes_temp = pd.concat(group_process_list)
        result_df = nodes_temp.groupby('id', sort=False).agg(lambda x: x.tolist())

        for col in nodes_raw.columns.tolist():
            if col == "case" or col == "id":
                continue
            node_feature = result_df[col].tolist()
            self.save_to_file(node_feature, f"{type}_{col}", fold)

    def save_to_file(self, data, filename, fold):
        with open(f"{self.data_dir}/{self.event_name}_kfoldcv_{fold}/{filename}.npy", 'wb') as file:
            pickle.dump(data, file)

    def load_from_file(self, filename, fold):
        with open(f"{self.data_dir}/{self.event_name}_kfoldcv_{fold}/{filename}.npy", 'rb') as file:
            return pickle.load(file)

    def process_data(self):
        for fold in tqdm(range(3), desc="Processing fold"):
            train_df, val_df, test_df = self.prepare_datasets(fold)
            train_df, val_df, test_df = self.process_timestamps(train_df, val_df, test_df)
            self.discretize_and_encode(train_df, val_df, test_df, fold)
            self.process_and_save_edges(train_df, val_df, test_df, fold)
            self.process_and_save_nodes_feature(train_df, val_df, test_df, fold)

    def prepare_datasets(self, fold):
        df_train = pd.read_csv(f"{self.three_fold_dir}/{self.event_name}_kfoldcv_{fold}_train.csv", sep=',', header=0, index_col=False)
        df_test = pd.read_csv(f"{self.three_fold_dir}/{self.event_name}_kfoldcv_{fold}_test.csv", sep=',', header=0, index_col=False)

        np.random.seed(133)
        grouped = df_train.groupby('case')
        new_order = np.random.permutation(list(grouped.groups.keys()))
        new_groups = [grouped.get_group(key) for key in new_order]
        log_shuffled = pd.concat(new_groups)
        log_shuffled.index = range(len(log_shuffled))
        train, valid = train_test_split(log_shuffled, test_size=0.2, shuffle=False)
        train.index = range(len(train))
        valid.index = range(len(valid))

        self.save_to_csv(train, f"{self.event_name}_train", fold)
        self.save_to_csv(valid, f"{self.event_name}_valid", fold)
        self.save_to_csv(df_test, f"{self.event_name}_test", fold)

        return train, valid, df_test

    def save_to_csv(self, df, filename, fold):
        os.makedirs(f"{self.data_dir}/{self.event_name}_kfoldcv_{fold}", exist_ok=True)
        df.to_csv(f"{self.data_dir}/{self.event_name}_kfoldcv_{fold}/{filename}.csv", index=False)

    def process_timestamps(self, train_df, val_df, test_df):
        train_df = DataProcessingTools.to_duration(train_df)
        val_df = DataProcessingTools.to_duration(val_df)
        test_df = DataProcessingTools.to_duration(test_df)
        return train_df, val_df, test_df

    def discretize_and_encode(self, train_df, val_df, test_df, fold):
        for col in train_df.columns.tolist():
            if col == "case":
                continue
            train_df[col].fillna(method='ffill', inplace=True)
            val_df[col].fillna(method='ffill', inplace=True)
            test_df[col].fillna(method='ffill', inplace=True)

            total_data = pd.concat([train_df, val_df, test_df])
            cont_trace = total_data['case'].value_counts(dropna=False)
            mean_trace = int(round(np.mean(cont_trace)))

            if col == 'duration':
                self.discretize_duration(train_df, val_df, test_df, mean_trace, col)
                total_data = pd.concat([train_df, val_df, test_df])
                cont_trace = total_data['case'].value_counts(dropna=False)
                mean_trace = int(round(np.mean(cont_trace)))

            att_encode_map = DataProcessingTools.encode_map(set(total_data[col].values))
            train_df[col] = train_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
            val_df[col] = val_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
            test_df[col] = test_df[col].apply(lambda e: att_encode_map.get(str(e), -1))

            att_count = len(set(total_data[col].values))
            self.save_to_file(att_count, f"{col}_info", fold)
            self.feature_list.append(col)

        self.save_to_file(mean_trace, 'mean_trace', fold)
        self.save_to_file(max(cont_trace), 'max_trace', fold)

    def discretize_duration(self, train_df, val_df, test_df, mean_trace, col):
        source = train_df[col].to_numpy().reshape(-1, 1)
        target1 = val_df[col].to_numpy().reshape(-1, 1)
        target2 = test_df[col].to_numpy().reshape(-1, 1)
        uniform = KBinsDiscretizer(n_bins=mean_trace, encode='ordinal', strategy='quantile')
        uniform.fit(source)
        train_df[col] = uniform.transform(source).astype(str)
        val_df[col] = uniform.transform(target1).astype(str)
        test_df[col] = uniform.transform(target2).astype(str)

    def process_and_save_edges(self, train_df, val_df, test_df, fold):
        train_edges_raw = self.edges_process(train_df)
        val_edges_raw = self.edges_process(val_df)
        test_edges_raw = self.edges_process(test_df)

        self.save_edges_feature(train_edges_raw, "train", fold)
        self.save_edges_feature(val_edges_raw, "val", fold)
        self.save_edges_feature(test_edges_raw, "test", fold)

    def process_and_save_nodes_feature(self, train_df, val_df, test_df, fold):
        train_nodes_raw = self.add_attribute_process(train_df)
        val_nodes_raw = self.add_attribute_process(val_df)
        test_nodes_raw = self.add_attribute_process(test_df)

        all_nodes_raw = pd.concat([train_nodes_raw, val_nodes_raw, test_nodes_raw])
        repeat_count = len(all_nodes_raw["repeat"].unique())
        occurrences_count = len(all_nodes_raw["occurrences"].unique())
        max_repeat_result = all_nodes_raw["occurrences"].max()

        self.save_to_file(repeat_count, 'repeat_info', fold)
        self.save_to_file(occurrences_count, 'occurrences_info', fold)
        self.save_to_file(self.feature_list, 'feature_list', fold)
        self.save_to_file(max_repeat_result, "max_repeat_result", fold)

        self.save_node_feature(train_nodes_raw, "train", fold)
        self.save_node_feature(val_nodes_raw, "val", fold)
        self.save_node_feature(test_nodes_raw, "test", fold)


if __name__ == '__main__':
    event_name = args.dataset
    processor = EventLogProcessor(event_name=event_name)
    processor.process_data()
