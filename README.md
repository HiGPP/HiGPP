# **HiGPP: A History-informed Graph-based Process Predictor for Next Activity using Graph Convolutional Network**

This repository contains the source code and related files for the paper titled "Predictive Business Process Monitoring based on History-informed Graph using Graph Convolutional Network". The datasets are located in the raw_dir/three_fold_data directory.

## - **How to Use**

- Specify the preprocessed event log for extracting graph information to create a History-informed Graph (example specifies the dataset as bpi13_problems)

  - ```
    python3 HistoryInformGraph.py -d bpi13_problems
    ```

- To train the model, run: (example specifies the dataset as bpi13_problems, fold is specified for three-fold cross-validation)

  - ```
    python3 train.py -d bpi13_problems --hidden-dim 128 --num-layers 2 --num-epochs 50 --batch-size 64 --gpu 0 --fold 0
    ```

  - ```
    python3 train.py -d bpi13_problems --hidden-dim 128 --num-layers 2 --num-epochs 50 --batch-size 64 --gpu 0 --fold 1
    ```

  - ```
    python3 train.py -d bpi13_problems --hidden-dim 128 --num-layers 2 --num-epochs 50 --batch-size 64 --gpu 0 --fold 2
    ```

- To load the saved trained model and evaluate it on the test set, run: (example specifies the dataset as bpi13_problems, fold is specified for three-fold cross-validation)

  - ```
    python3 test.py -d bpi13_problems --hidden-dim 128 --num-layers 2 --num-epochs 50 --batch-size 64 --gpu 0 --fold 0
    ```

  - ```
    python3 test.py -d bpi13_problems --hidden-dim 128 --num-layers 2 --num-epochs 50 --batch-size 64 --gpu 0 --fold 1
    ```

  - ```
    python3 test.py -d bpi13_problems --hidden-dim 128 --num-layers 2 --num-epochs 50 --batch-size 64 --gpu 0 --fold 2
    ```

# - **Environment**

- PyTorch version: pytorch1.13.0+cu116
