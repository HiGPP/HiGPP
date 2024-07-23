import os

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, average_precision_score
import pickle
import sys
from tensorflow.keras.models import load_model
from imblearn.metrics import geometric_mean_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

def Union(lst1, lst2):
    final_list = lst1 + lst2
    return final_list

def multiclass_roc_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)

def multiclass_pr_auc_score(y_test, y_pred, average):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return average_precision_score(y_test, y_pred, average=average)


if __name__ == '__main__':
    eventlog = sys.argv[1]
    

    for fold in range(3):
        X_a_test = np.load("image/" + eventlog + f'_{fold}'+ "/" + eventlog + "_test.npy")
        X_a_test = X_a_test / 255.0

        
        outfile2 = open(eventlog+".txt", 'a')

        model = load_model("model/"+eventlog  +f'_{fold}'+".h5")

        with open("image/" + eventlog  +f'_{fold}'+ "/" + eventlog + "_train_y.pkl", 'rb') as handle:
            y_train = pickle.load(handle)

        with open("image/" + eventlog  +f'_{fold}'+ "/" + eventlog + "_test_y.pkl", 'rb') as handle:
            y_test = pickle.load(handle)

        df_labels = np.unique(list(y_train) + list(y_test))

        label_encoder = preprocessing.LabelEncoder()
        integer_encoded = label_encoder.fit_transform(df_labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
        onehot_encoder.fit(integer_encoded)
        onehot_encoded = onehot_encoder.transform(integer_encoded)

        train_integer_encoded = label_encoder.transform(y_train).reshape(-1, 1)
        train_onehot_encoded = onehot_encoder.transform(train_integer_encoded)
        Y_train = np.asarray(train_onehot_encoded)

        test_integer_encoded = label_encoder.transform(y_test).reshape(-1, 1)
        test_onehot_encoded = onehot_encoder.transform(test_integer_encoded)
        Y_test = np.asarray(test_onehot_encoded)
        Y_test_int = np.asarray(test_integer_encoded)

        n_classes = len(df_labels)
        preds_a = model.predict(X_a_test)

        y_a_test = np.argmax(Y_test, axis=1)
        preds_a = np.argmax(preds_a, axis=1)

        precision, recall, fscore, _ = precision_recall_fscore_support(Y_test_int, preds_a, average='macro',
                                                                       pos_label=None)
        auc_score_macro = multiclass_roc_auc_score(Y_test_int, preds_a, average="macro")
        prauc_score_macro = multiclass_pr_auc_score(Y_test_int, preds_a, average="macro")

        g_mean = geometric_mean_score(Y_test_int, preds_a, average="macro")

        print(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write(classification_report(Y_test_int, preds_a, digits=3))
        outfile2.write('\nAUC: ' + str(auc_score_macro))
        outfile2.write('\nPRAUC: '+ str(prauc_score_macro))
        outfile2.write('\n')
        outfile2.write('\nGMEAN: ' + str(round(g_mean, 3)))
        outfile2.write('\n' )


        outfile2.flush()

        outfile2.close()
