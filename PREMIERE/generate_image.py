import numpy as np
import pandas as pd
from sklearn import preprocessing
import argparse

from tqdm import tqdm


def dec_to_bin(x):
    return format(int(x), "b")


def rgb_image_generation(df, num_col):
    list_image_flat = []
    size, padding = get_image_size(num_col)
    vec = [[0, 0, 0]] * padding
    for (index_label, row_series) in df.iterrows():
        list_image = []
        j = 0
        while j < len(row_series):
            if row_series[j] > 1:
                c = 1.0
            elif row_series[j] < 0:
                c = 0.0
            else:
                c = row_series[j]
            v = c * (2 ** 24 - 1)
            bin_num = dec_to_bin(int(v))
            if len(bin_num) < 24:
                pad = 24 - len(bin_num)
                zero_pad = "0" * pad
                line = zero_pad + str(bin_num)
                n = 8
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
                int_num = [int(element, 2) for element in rgb]
            else:
                n = 8
                line = str(bin_num)
                rgb = [line[i:i + n] for i in range(0, len(line), n)]
                int_num = [int(element, 2) for element in rgb]
            list_image.append(int_num)
            j = j + 1
        list_image = list_image + vec
        new_img = np.asarray(list_image)
        new_img = new_img.reshape(size, size, 3)
        list_image_flat.append(new_img)
    return list_image_flat


def get_image_size(num_col):
    import math
    matx = round(math.sqrt(num_col))
    if num_col > (matx * matx):
        matx = matx + 1
        padding = (matx * matx) - num_col
    else:
        padding = (matx * matx) - num_col
    return matx, padding


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
        'bpi13_closed_problems'
    ]

    for eventlog in tqdm(list_eventlog):
        for fold in range(3):
            namedataset = eventlog
            print(f"\nprocess: {eventlog}")

            df_train = pd.read_csv("feature_fold/" + namedataset + f'_{fold}' + '/' + namedataset + '_train.csv',
                                   header=None)
            df_test = pd.read_csv("feature_fold/" + namedataset + f'_{fold}' + '/' + namedataset + '_test.csv',
                                  header=None)

            df_train = df_train.iloc[:, :-1]
            df_test = df_test.iloc[:, :-1]

            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            scaler.fit(df_train.values.astype(float))

            train_norm = scaler.transform(df_train.values.astype(float))
            test_norm = scaler.transform(df_test.values.astype(float))

            col = train_norm.shape[1]

            train_norm = pd.DataFrame(train_norm)
            test_norm = pd.DataFrame(test_norm)

            X_train = rgb_image_generation(train_norm, col)
            X_test = rgb_image_generation(test_norm, col)

            np.save("image/" + namedataset + f'_{fold}' + "/" + namedataset + "_train" + ".npy", X_train)
            np.save("image/" + namedataset + f'_{fold}' + "/" + namedataset + "_test" + ".npy", X_test)
