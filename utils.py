import pandas as pd

class DataProcessingTools:
    @staticmethod
    def to_duration(df):
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
        groupProcessList = []

        for groupId, group in df.groupby('case', sort=False):
            group['year'] = group['timestamp'].dt.year
            group['month'] = group['timestamp'].dt.month
            group['day'] = group['timestamp'].dt.date
            group['day_of_week'] = group['timestamp'].dt.day_of_week
            group['period'] = group['timestamp'].dt.strftime('%p')
            group['hour'] = group['timestamp'].dt.hour
            group['is_weekend'] = (group['day_of_week'] >= 5).astype(int)
            group['duration'] = group['timestamp'].diff().dt.total_seconds().fillna(0)
            groupProcessList.append(group)

        df = pd.concat(groupProcessList).sort_index()
        return df.drop('timestamp', axis=1)

    @staticmethod
    def encode_map(input_array):
        p_map = {}
        length = len(input_array)
        for index, ele in zip(range(1, length + 1), input_array):
            p_map[str(ele)] = index
        return p_map

    @staticmethod
    def decode_map(encode_map):
        de_map = {}
        for k, v in encode_map.items():
            de_map[v] = k
        return de_map

    @staticmethod
    def get_prefix_sequence(sequence):
        i = 0
        list_seq = []
        while i < len(sequence):
            list_temp = []
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                list_seq.append(list(list_temp))
                j = j + 1
            i = i + 1
        return list_seq

    @staticmethod
    def get_prefix_sequence_label(sequence):
        i = 0
        list_seq = []
        list_label = []
        list_len = []
        while i < len(sequence):
            list_temp = []
            j = 0
            while j < (len(sequence.iat[i, 0]) - 1):
                list_temp.append(sequence.iat[i, 0][0 + j])
                list_seq.append(list(list_temp))
                list_label.append(sequence.iat[i, 0][j + 1])
                list_len.append(j + 1)
                j = j + 1
            i = i + 1
        return list_seq, list_label, list_len
