# **HiGPP: A History-informed Graph-based Process Predictor for Next Activity using Graph Convolutional Network**

This repository contains the source code and related files for the paper titled "Predictive Business Process Monitoring based on History-informed Graph using Graph Convolutional Network". The datasets are located in the raw_dir/three_fold_data directory.

## - **How to Use**

- Specify the preprocessed event log for extracting graph information to create a History-informed Graph (example specifies the dataset as bpi13_problems)

  - ```
    python3 HistoryInformGraph.py -d bpi13_problems --fold 0
    python3 HistoryInformGraph.py -d bpi13_problems --fold 1
    python3 HistoryInformGraph.py -d bpi13_problems --fold 2
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

# - **Baselines**

This repository also includes implementations of various baseline methods for predictive business process monitoring, including MiDA, ProcessTransformer, MiTFM, PREMIERE, gcn-procesprediction, BIGDGCNN, and Multi-BIGDGCNN.

## Baseline Methods

### MiDA
- **Reference:** Vincenzo Pasquadibisceglie, Annalisa Appice, Giovanna Castellano, and Donato Malerba. A multi-view deep learning approach for predictive business process monitoring. IEEE Transactions on Services Computing, 15(4):2382–2395, 2022.

### ProcessTransformer
- **Reference:** Zaharah A Bukhsh, Aaqib Saeed, and Remco M Dijkman. Processtransformer: Predictive business process monitoring with transformer network. arXiv preprint arXiv:2104.00721, 2021.

### MiTFM
- **Reference:** Jiaxing Wang, Chengliang Lu, Bin Cao, and Jing Fan. MiTFM: A multi-view information fusion method based on transformer for next activity prediction of business processes. In Proceedings of the 14th Asia-Pacific Symposium on Internetware, pages 281–291, 2023.

### PREMIERE
- **Reference:** V. Pasquadibisceglie, A. Appice, G. Castellano, and D. Malerba. Predictive process mining meets computer vision. In Business Process Management Forum: BPM Forum 2020, Seville, Spain, September 13–18, 2020, Proceedings 18, pages 176–192, 2020.

### GCN-ProcessPrediction
- **Reference:** Ishwar Venugopal, Jessica Töllich, Michael Fairbank, and Ansgar Scherp. A comparison of deep-learning methods for analysing and predicting business processes. In 2021 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2021.

### BIGDGCNN
- **Reference:** Andrea Chiorrini, Claudia Diamantini, Alex Mircoli, and Domenico Potena. Exploiting instance graphs and graph neural networks for next activity prediction. In International conference on process mining, pages 115–126, 2021.

### Multi-BIGDGCNN
- **Reference:** Andrea Chiorrini, Claudia Diamantini, Laura Genga, and Domenico Potena. Multi-perspective enriched instance graphs for next activity prediction through graph neural network. Journal of Intelligent Information Systems, 61(1):5–25, 2023.
