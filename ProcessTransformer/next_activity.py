import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, average_precision_score
import tensorflow as tf
from sklearn import metrics

from processtransformer import constants
from processtransformer.data import loader
from processtransformer.models import transformer

parser = argparse.ArgumentParser(description="Process Transformer - Next Activity Prediction.")

parser.add_argument("--dataset", default="bpi13_closed_problems", type=str, help="dataset name")

parser.add_argument("--model_dir", default="./models", type=str, help="model directory")

parser.add_argument("--result_dir", default="./results", type=str, help="results directory")

parser.add_argument("--task", type=constants.Task, 
    default=constants.Task.NEXT_ACTIVITY,  help="task name")
parser.add_argument("--fold", default=2, type=int, help="fold")

parser.add_argument("--epochs", default=10, type=int, help="number of total epochs")

parser.add_argument("--batch_size", default=12, type=int, help="batch size")

parser.add_argument("--learning_rate", default=0.01, type=float,
                    help="learning rate")

parser.add_argument("--gpu", default=0, type=int, 
                    help="gpu id")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


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


if __name__ == "__main__":
    
    model_path = f"{args.model_dir}/{args.dataset}"
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = f"{model_path}/next_activity_ckpt"

    result_path = f"{args.result_dir}/{args.dataset}"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    result_path = f"{result_path}/results"

    
    data_loader = loader.LogsDataLoader(name = args.dataset, fold = args.fold)

    (train_df, test_df, x_word_dict, y_word_dict, max_case_length, 
        vocab_size, num_output) = data_loader.load_data(args.task)
    
    
    train_token_x, train_token_y = data_loader.prepare_data_next_activity(train_df, 
        x_word_dict, y_word_dict, max_case_length)
    
    
    transformer_model = transformer.get_next_activity_model(
        max_case_length=max_case_length, 
        vocab_size=vocab_size,
        output_dim=num_output)

    transformer_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        save_weights_only=True,
        monitor="sparse_categorical_accuracy",
        mode="max", save_best_only=True)


    transformer_model.fit(train_token_x, train_token_y, 
        epochs=args.epochs, batch_size=args.batch_size, 
        shuffle=True, verbose=2, callbacks=[model_checkpoint_callback])

    
    
    
    test_data_subset = test_df  
    if len(test_data_subset) > 0:
        test_token_x, test_token_y = data_loader.prepare_data_next_activity(test_data_subset,
            x_word_dict, y_word_dict, max_case_length)
        y_pred = np.argmax(transformer_model.predict(test_token_x), axis=1)
        result_path = "results/" + args.dataset
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        outfile2 = open(result_path + "/" + args.dataset + "_" + ".txt", 'a')
        precision, recall, fscore, _ = precision_recall_fscore_support(test_token_y, y_pred, average='macro',
                                                                       pos_label=None)

        auc_score_macro = multiclass_roc_auc_score(test_token_y, y_pred, average="macro")
        prauc_score_macro = multiclass_pr_auc_score(test_token_y, y_pred, average="macro")

        print(classification_report(test_token_y, y_pred, digits=3))
        outfile2.write(classification_report(test_token_y, y_pred, digits=3))
        outfile2.write('\nAUC: ' + str(auc_score_macro))
        outfile2.write('\nPRAUC: ' + str(prauc_score_macro))
        outfile2.write('\n')

        outfile2.flush()
        outfile2.close()

            
            
            
            
            

    
    
    
    
    
    
    
    
    
    
    
    