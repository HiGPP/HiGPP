from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from hyperopt import Trials, STATUS_OK, tpe, fmin, hp
import hyperopt
from time import perf_counter
import time
import os
from sklearn import preprocessing
import argparse
import pickle
import random

import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_visible_devices([gpus[0]], 'GPU')

tf.config.experimental.set_memory_growth(gpus[0], True)

os.environ['PYTHONHASHSEED'] = '0'
seed = 42
tf.random.set_seed(seed)

np.random.seed(seed)

random.seed(seed)


def get_image_size(num_col):
    import math
    matx = round(math.sqrt(num_col))
    if num_col > (matx * matx):
        matx = matx + 1
        padding = (matx * matx) - num_col
    else:
        padding = (matx * matx) - num_col
    return matx, padding


def inception_module(layer_in, f1, f2, f3):
    conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)

    conv3 = Conv2D(f2, (3, 3), padding='same', activation='relu')(layer_in)

    conv5 = Conv2D(f3, (5, 5), padding='same', activation='relu')(layer_in)

    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)

    layer_out = concatenate([conv1, conv3, conv5, pool], axis=-1)
    return layer_out


def get_model(dense1, dense2, dropout1, dropout2, n_classes, learning_rate):
    inputs = Input(shape=(img_size, img_size, 3))
    filters = (inception_module(inputs, f1, f2, f3))
    filters = (inception_module(filters, f1, f2, f3))

    layer_out = Dense(dense1, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))(
        filters)
    layer_out = Dropout(dropout1)(layer_out)
    layer_out = Flatten()(layer_out)
    layer_out = Dense(dense2, activation='relu', kernel_initializer=tf.keras.initializers.glorot_uniform(seed=seed))(
        layer_out)
    layer_out = Dropout(dropout2)(layer_out)

    optimizer = Adam(learning_rate=learning_rate)

    out = Dense(n_classes, activation='softmax')(layer_out)
    model = Model(inputs=inputs, outputs=out)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
    model.summary()

    return model


def fit_and_score(params):
    print(params)
    start_time = perf_counter()

    model = get_model(learning_rate=params['learning_rate'], dense1=params['dense1'], dense2=params['dense2'],
                      dropout1=params['dropout1'], dropout2=params['dropout2'], n_classes=params['n_classes'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    h = model.fit(X_a_train, train_onehot_encoded, epochs=200, verbose=0, validation_split=0.2,
                  callbacks=[early_stopping], batch_size=2 ** params['batch_size'])

    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)
    print(score)

    global best_score, best_model, best_time, best_numparameters
    end_time = perf_counter()

    if best_score > score:
        best_score = score
        best_model = model
        best_numparameters = model.count_params()
        best_time = end_time - start_time

    return {'loss': score, 'status': STATUS_OK, 'n_epochs': len(h.history['loss']), 'n_params': model.count_params(),
            'time': end_time - start_time}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inception for next activity prediction.')
    parser.add_argument('-event_log', type=str, help="Event log name")
    args = parser.parse_args()
    namedataset = args.event_log

    for fold in range(3):


        output_file = namedataset

        current_time = time.strftime("%d.%m.%y-%H.%M", time.localtime())
        outfile = open(output_file + '.log', 'w')

        outfile.write("Starting time: %s\n" % current_time)

        n_iter = 20
        f1, f2, f3 = 64, 128, 32

        df_train = pd.read_csv("feature_fold/" + namedataset + f'_{fold}' + '/' + namedataset + '_train.csv',
                               header=None)
        df_test = pd.read_csv("feature_fold/" + namedataset + f'_{fold}' + '/' + namedataset + '_test.csv',
                              header=None)

        with open("image/" + namedataset + f'_{fold}' + "/" + namedataset + "_train_y.pkl", 'rb') as handle:
            l_train = pickle.load(handle)

        with open("image/" + namedataset + f'_{fold}' + "/" + namedataset + "_test_y.pkl", 'rb') as handle:
            l_test = pickle.load(handle)

        df_labels = np.unique(list(l_train) + list(l_test))
        n_classes = len(df_labels)

        label_encoder = preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(df_labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder.fit(integer_encoded)
        onehot_encoded = onehot_encoder.transform(integer_encoded)

        train_integer_encoded = label_encoder.transform(l_train).reshape(-1, 1)
        train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
        Y_train = np.asarray(train_onehot_encoded)

        test_integer_encoded = label_encoder.transform(l_test).reshape(-1, 1)
        test_onehot_encoded = onehot_encoder.transform(test_integer_encoded)
        Y_test = np.asarray(test_onehot_encoded)
        Y_test_int = np.asarray(test_integer_encoded)

        space = {'dense1': hp.choice('dense1', [32, 64, 128]),
                 'dense2': hp.choice('dense2', [32, 64, 128]),
                 'dropout1': hp.uniform("dropout1", 0, 1),
                 'dropout2': hp.uniform("dropout2", 0, 1),
                 'batch_size': hp.choice('batch_size', [5, 6, 7]),
                 'learning_rate': hp.loguniform("learning_rate", np.log(0.00001), np.log(0.01)),
                 'n_classes': n_classes}

        X_a_train = np.load("image/" + namedataset + f'_{fold}' + "/" + namedataset + "_train.npy")
        X_a_test = np.load("image/" + namedataset + f'_{fold}' + "/" + namedataset + "_test.npy")
        img_size = X_a_train.shape[1]

        X_a_train = np.asarray(X_a_train)
        X_a_test = np.asarray(X_a_test)

        X_a_train = X_a_train.astype('float32')
        X_a_train = X_a_train / 255.0

        X_a_test = X_a_test.astype('float32')
        X_a_test = X_a_test / 255.0

        print('Starting model selection...')
        best_score = np.inf
        best_model = None
        best_time = 0
        best_numparameters = 0

        trials = Trials()

        best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=n_iter, trials=trials,
                    rstate=np.random.default_rng(seed))

        best_params = hyperopt.space_eval(space, best)

        outfile.write("\nHyperopt trials")
        outfile.write("\ntid,loss,learning_rate,batch_size,time,n_epochs,n_params,perf_time,dense1,dense2,drop1,drop2")
        for trial in trials.trials:
            outfile.write("\n%d,%f,%f,%d,%s,%d,%d,%f,%d,%d,%f,%f" % (trial['tid'],
                                                                     trial['result']['loss'],
                                                                     trial['misc']['vals']['learning_rate'][0],
                                                                     trial['misc']['vals']['batch_size'][0] + 7,
                                                                     (trial['refresh_time'] - trial[
                                                                         'book_time']).total_seconds(),
                                                                     trial['result']['n_epochs'],
                                                                     trial['result']['n_params'],
                                                                     trial['result']['time'],
                                                                     trial['misc']['vals']['dense1'][0],
                                                                     trial['misc']['vals']['dense2'][0],
                                                                     trial['misc']['vals']['dropout1'][0],
                                                                     trial['misc']['vals']['dropout2'][0]
                                                                     ))
        outfile.write("\n\nBest parameters:")
        print(best_params, file=outfile)
        outfile.write("\nModel parameters: %d" % best_numparameters)
        outfile.write('\nBest Time taken: %f' % best_time)

        save_path = 'model/' + namedataset + f'_{fold}' + '.h5'
        save_dir = os.path.dirname(save_path)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        best_model.save(save_path)

        outfile.flush()

        outfile.close()
