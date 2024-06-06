# Utilisation of machine learning algorithms incorporating Laplacian and unsupervised methods for the detection of enzymes from protein graphs
This repository contains the report and code for the implementation of a project that seeks to investigate the use of different ML methods for the detection of enzymes from protein graphs. 

To read about the project in detail, please consult the report attached. The implementation code is provided in a Python notebook.

# Authors and Contributors
1. Jyotishka Das
2. Arun Manivannan
3. Victor Hong
4. Hao Xu

# Abstract
Graph classification in network analysis involves predicting the class label of a graph based on its structure. A graph is defined by a set of nodes and edges that represent a complex
system, for example social networks, computer networks, biological pathways, or molecular structures. Graph classification has been addressed using various methods, including but not limited to, graph kernels, neural networks, and random walks. In all cases, the objective is to arrive at accurate and efficient algorithms that accurately map graph structures that represent large-scale datasets with a variety of
structures. 

In our examination of literature on the subject, we were keen to explore efficient yet accurate algorithms to perform graph classification, and we arrived at incorporating the use of the spectral decomposition of graph Laplacian to perform graph classification. In summarised terms, the graph Laplacian is a matrix that captures the connectivity and structure of a graph. Eigenvalues and eigenvectors of the Laplacian matrix are used to decompose a graph into spectral components. Each eigenvector corresponds to a specific frequency of the graph that can be used to identify substructures within the
graph. By selecting a subset of these eigenvectors, the dimensionality of the graph can be reduced and used as inputfeatures for machine learning models.

# Motivation
Biological structures can be visualized as a graph using nodes and edges. Nodes can be molecular structures (atoms) and edges can represent the connections between these structures through chemical bonds or spatial relationships. Recent advances in machine learning has enabled us to use graph learning methods to accurately classify biological structures, specifically chemical compounds. Examples of these chemical structures include molecular structures, proteins and enzymes.

A potential application of being able to classify molecular structures is to predict if a protein is enzyme or non-enzyme from its graphical representation. Our goal in this project is to determine if incorporating embedding techniques using Laplacian eigenvalues or random walk based methods would improve the accuracy of detection of enzymes using the Protein Full (PF) dataset, while achieving stable results across a range of machine learning techniques to prove its reliability.

# Dataset
The publicly available Proteins Full (PF) network dataset was used for this work (Rossi, R.A., Ahmed, N.K.: The network data repository
with interactive graph analytics and visualization. In: AAAI, https://networkrepository.com, 2015).

# Methodology
The methodology of the project revolves around two major steps, namely obtaining the embeddings from the graph and then use a classifier to obtain the final accuracy results. We studied two methods to obtain the embeddings from the graph. The first method is a Laplacian spectral-based method, and the second one is an unsupervised random walk based node2vec method.

![Algorithm](https://github.com/dasjyotishka/Utilisation-of-machine-learning-algorithms-for-the-detection-of-enzymes-from-protein-graphs/assets/55792433/008f290f-df5a-4881-901e-9cb31b6a60aa)


# Results
A comparison of the classification accuracy obtained using different approaches are tabulated in Table 1. The rows headers of the table correspond to the algirithms used to generate the embeddings while the column headers correspond to the choice of the machine learning classifier.

  #### Table 1. Comparison of the classification accuracy obtained using different approaches

| Model         | Laplacian Eigenvalues | Node2vec Embeddings |
|---------------|----------|-----------------------|
| Random Forest | 0.75     | 0.68                  | 
| XGBoost       | 0.72     | 0.67                  |     
| CatBoost      | 0.73     | 0.69                  |                     
| SVM           | 0.75     | 0.69                  |                     
| LightGBM      | 0.73     | 0.66                  |                     


We have tried to benchmark our model for the detection of enzymes from protein graphs against many recent state-ofthe- arts deep learning models utilising Graph neural network methods on the same Protein Full (PF) dataset in Table 2. 

  #### Table 2. Comparison of the performance of the classification accuracy with recent “deep learning" based graphical models on the Protein Full dataset
  | Algorithm                                          | Accuracy | Reference               |
|----------------------------------------------------|----------|-------------------------|
| Graph Isomorphism Network (GIN)                    | 0.84     | Xu et al.      |
| Deep Graph Convolutional Neural Network (DGCNN)    | 0.81     | Zhang et al.     |
| Graph Convolutional Network (GCN)                  | 0.76     | Kipf & Welling.  |
| Graph Attention Network (GAT)                      | 0.75     | Velickovic et al.|
| Laplacian embeddings with Random Forest/SVM        | 0.75     | Proposed Method         |

  
  The proposed method gives comparable performance in terms of accuracy when compared with a Graph Attention Network (GAN) or a Graph Convolutional Network (GCN), whereas more complex methods like Graph Isomorphism Network (GIN) or Deep Graph Convolutional Neural Network (DGCNN) gives a better classification accuracy. On the contrary, these deep networks take huge computation time to train their networks.We were unable to find a comparison of the time taken to train the models in the available literatures as a means of benchmarking, but available literatures show that these deep learning algorithms take considerable time for training owing to the repeated number of trials to arrive at the right set of hyperparameters. Also, these deeper networks extract the embeddings from the graph automatically, and as such it puts a question on their human interpretability.


