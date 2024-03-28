import warnings
from args import args
from tqdm import tqdm
warnings.filterwarnings(action='ignore')
from utils import *
import pickle

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
import os

def edges_process(event_log):
    groupProcessList = []
    for groupId, group in event_log.groupby('case', sort=False):
        group = group.copy()
        group = pd.concat([group.iloc[0:1], group])
        group["activity:next"] = group["activity"]
        group["activity:next"] = group["activity:next"].shift(-1)
        group = group.iloc[:-1, :]
        group = group.loc[:, ["case", "activity", "activity:next"]]
        groupProcessList.append(group)
    edges_raw = pd.concat(groupProcessList)
    edges_raw.rename(columns={"activity": "source",
                              "activity:next": "destination"}, inplace=True)
    edges_raw['source'] = edges_raw['source'].astype(int)
    edges_raw['destination'] = edges_raw['destination'].astype(int)
    return edges_raw

def add_attribute_process(event_log):
    groupProcessList = []
    for groupId, group in event_log.groupby('case', sort=False):
        group = group.copy()
        group['occurrences'] = group.groupby("activity", sort=False).cumcount() + 1
        group['repeat'] = group['occurrences'].map(lambda x: 2 if x > 1 else 1)
        groupProcessList.append(group)
    add_attribute_raw = pd.concat(groupProcessList)
    return add_attribute_raw

def save_edges_feature(raw, type, f):
    for col in raw.columns.tolist():
        if col == "case":
            continue
        else:
            sequence = raw.groupby("case", sort=False).agg({col: lambda x: list(x)})
            if col == "destination":
                list_seq, list_label, list_len = get_prefix_sequence_label(sequence)
                with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + type + '_' + col + ".npy", 'wb') as file:
                    pickle.dump(list_seq, file)
                with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + type + '_' + "label" + ".npy", 'wb') as file:
                    pickle.dump(list_label, file)
                with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + type + '_' + "len" + ".npy", 'wb') as file:
                    pickle.dump(list_len, file)
            else:
                list_seq = get_prefix_sequence(sequence)
                with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + type + '_' + col + ".npy", 'wb') as file:
                    pickle.dump(list_seq, file)


def save_node_feature(nodes_raw, type, f):
    with open("raw_dir/" + event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + type + '_' + "source" + ".npy",
              'rb') as file:
        source_lists = pickle.load(file)
    with open("raw_dir/" + event_name + "/" + event_name + "_kfoldcv_" + str(
            f) + "/" + type + '_' + "destination" + ".npy", 'rb') as file:
        destination_lists = pickle.load(file)
    groupProcessList = []
    index = 0
    new_source_lists = []
    new_destination_lists = []
    for groupId, group in nodes_raw.groupby('case', sort=False):
        i = 0
        while i < len(group) - 1:
            i = i + 1
            group_temp = group[:i]
            unique_values = group_temp['activity'].drop_duplicates(keep='last')
            df_activity = pd.DataFrame(unique_values)
            group_temp = group_temp.groupby('activity', sort=False).agg(
                lambda x: x.tolist()).reset_index()
            group_temp = pd.merge(df_activity, group_temp, on='activity', how='left')
            group_temp['id'] = index
            groupProcessList.append(group_temp)
            encoding_map = {value: code for code, value in enumerate(unique_values)}
            source_list = [encoding_map[value] for value in source_lists[index]]
            new_source_lists.append(source_list)
            destination_list = [encoding_map[value] for value in destination_lists[index]]
            new_destination_lists.append(destination_list)
            index = index + 1
    with open("raw_dir/" + event_name + "/" + event_name + "_kfoldcv_" + str(
            f) + "/" + type + '_' + "new_source" + ".npy", 'wb') as file:
        pickle.dump(new_source_lists, file)
    with open("raw_dir/" + event_name + "/" + event_name + "_kfoldcv_" + str(
            f) + "/" + type + '_' + "new_destination" + ".npy", 'wb') as file:
        pickle.dump(new_destination_lists, file)
    nodes_temp = pd.concat(groupProcessList)
    result_df = nodes_temp.groupby('id', sort=False).agg(lambda x: x.tolist())
    for col in nodes_raw.columns.tolist():
        if col == "case" or col == "id":
            continue
        else:
            node_feature = result_df[col].tolist()
            with open(
                    "raw_dir/" + event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + type + '_' + col + ".npy",
                    'wb') as file:
                pickle.dump(node_feature, file)

if __name__ == '__main__':
    event_name = args.dataset
    for f in tqdm(range(3), desc="Processing fold"):
        df_train = pd.read_csv("raw_dir/three_fold_data/" + event_name + "/" + event_name + "_kfoldcv_" + str(f) + "_train.csv",
                               sep=',',
                               header=0, index_col=False)
        df_test = pd.read_csv("raw_dir/three_fold_data/" + event_name + "/" + event_name + "_kfoldcv_" + str(f) + "_test.csv",
                              sep=',',
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
        if not os.path.exists("raw_dir/" + event_name + "/" + event_name + "_kfoldcv_" + str(f)):
            os.makedirs("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f))
        train.to_csv("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + event_name + "_train.csv", index=False)
        valid.to_csv("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + event_name + "_valid.csv", index=False)
        df_test.to_csv("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + event_name + "_test.csv", index=False)

        train_df = pd.read_csv("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + event_name + "_train.csv",
                               sep=',',
                               header=0, index_col=False)
        val_df = pd.read_csv("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + event_name + "_valid.csv",
                             sep=',',
                             header=0, index_col=False)
        test_df = pd.read_csv("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + event_name + "_test.csv",
                              sep=',',
                              header=0, index_col=False)

        train_df = to_duration(train_df)
        val_df = to_duration(val_df)
        test_df = to_duration(test_df)


        feature_list = []

        for col in train_df.columns.tolist():
            if col == "case":
                continue
            train_df[col].fillna(method='ffill', inplace=True)
            val_df[col].fillna(method='ffill', inplace=True)
            test_df[col].fillna(method='ffill', inplace=True)

            total_data = pd.concat([train_df, val_df, test_df])
            cont_trace = total_data['case'].value_counts(dropna=False)
            max_trace = max(cont_trace)
            mean_trace = int(round(np.mean(cont_trace)))
            if col == 'duration':
                source = train_df[col].to_numpy().reshape(-1, 1)
                target1 = val_df[col].to_numpy().reshape(-1, 1)
                target2 = test_df[col].to_numpy().reshape(-1, 1)
                uniform = KBinsDiscretizer(n_bins=mean_trace, encode='ordinal', strategy='quantile')
                uniform.fit(source)
                train_df[col] = uniform.transform(source).astype(str)
                val_df[col] = uniform.transform(target1).astype(str)
                test_df[col] = uniform.transform(target2).astype(str)
                total_data = pd.concat([train_df, val_df, test_df])
                cont_trace = total_data['case'].value_counts(dropna=False)
                max_trace = max(cont_trace)
                mean_trace = int(round(np.mean(cont_trace)))

            att_encode_map = encode_map(set(total_data[col].values))
            train_df[col] = train_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
            val_df[col] = val_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
            test_df[col] = test_df[col].apply(lambda e: att_encode_map.get(str(e), -1))
            # 记录特征的类别数
            att_count = len(set(total_data[col].values))
            with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + col + '_' + "info" + ".npy", 'wb') as file:
                pickle.dump(att_count, file)
            feature_list.append(col)

        # 保存平均轨迹长度
        with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + 'mean' + '_' + "trace" + ".npy", 'wb') as file:
            pickle.dump(mean_trace, file)
        with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + 'max' + '_' + "trace" + ".npy", 'wb') as file:
            pickle.dump(max_trace, file)

        train_edges_raw = edges_process(train_df)
        val_edges_raw = edges_process(val_df)
        test_edges_raw = edges_process(test_df)

        train_nodes_raw = add_attribute_process(train_df)
        val_nodes_raw = add_attribute_process(val_df)
        test_nodes_raw = add_attribute_process(test_df)

        all_nodes_raw = pd.concat([train_nodes_raw, val_nodes_raw, test_nodes_raw])
        repeat_count = len(all_nodes_raw["repeat"].unique())
        occurrences_count = len(all_nodes_raw["occurrences"].unique())
        max_repeat_result = all_nodes_raw["occurrences"].max()



        with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + 'repeat' + '_' + "info" + ".npy", 'wb') as file:
            pickle.dump(repeat_count, file)
        with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + 'occurrences' + '_' + "info" + ".npy", 'wb') as file:
            pickle.dump(occurrences_count, file)
        with open("raw_dir/"+ event_name + "/" + event_name + "_kfoldcv_" + str(f) + "/" + 'feature' + '_' + "list" + ".npy", 'wb') as file:
            pickle.dump(feature_list, file)

        with open("raw_dir/" + event_name + "/" + event_name + "_kfoldcv_" + str(
                f) + "/" + "max_repeat_result" + ".npy", 'wb') as file:
            pickle.dump(max_repeat_result, file)


        save_edges_feature(train_edges_raw, "train", f)
        save_edges_feature(val_edges_raw, "val", f)
        save_edges_feature(test_edges_raw, "test", f)

        save_node_feature(train_nodes_raw, "train", f)
        save_node_feature(val_nodes_raw, "val", f)
        save_node_feature(test_nodes_raw, "test", f)

