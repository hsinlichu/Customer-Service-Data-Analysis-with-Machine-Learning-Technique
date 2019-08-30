# Customer Service Data Analysis with Machine Learning Technique 
This project was done when I interned at [Cyberlink](https://www.cyberlink.com/index_en_US.html). I was responsible for analyzing with manchine learning technique. 

During my summer intern, I do few things below:
1. Supervised text classification
2. Unsupervised text clustering with different sentence representation and clustering algorithm.

    Sentence representation
    * Doc2Vec
    * TF-IDF
    
    Clustering Algorithm
    * Kmeans
    * DBSCAN
3. Topic Modeling - LDA
4. Find related question based on BERT.


## Dataset
This customer service data is around 0.1 million customer feedback sentence with subject, question sentence, and user selected question type.

![](https://i.imgur.com/2UfzcsT.png)



## Supervised text classification

### How to run
* Preprocessing: `python ./src/preprocess.py ./model/origin`
* Training: `python ./src/train.py ./model/origin`
* Run tensorboard: `tensorboard --logdir tensorboard`

## Unsupervised text clustering
Sentence representation
* Doc2Vec
* TF-IDF

Clustering Algorithm
* Kmeans
* DBSCAN
## Topic Modeling - LDA
## Find related question based on BERT.

## File Tree
```
Cyberlink-Intern/
├── analysis
│   ├── Customer Service Data Clustering.ipynb
│   ├── data_analysis.ipynb
│   ├── Data Preprocessing.ipynb
│   ├── LSTM Predict Confusion Matrix.ipynb
│   ├── picture
│   │   ├── ConfusionMatrix_2.jpg
│   │   ├── Confusion Matrix.jpg
│   │   ├── ConfusionMatrix.jpg
│   │   ├── ConfusionMatrix_newdata.jpg
│   │   ├── Confusion Matrix.png
│   │   ├── ConfusionMatrix_RelatedQuestion.png
│   │   ├── data_distribution.jpg
│   │   ├── data_distribution_ordered.jpg
│   │   ├── Normalized_ConfusionMatrix_2.jpg
│   │   ├── Normalized_ConfusionMatrix_newdata.jpg
│   │   └── Normalized_ConfusionMatrix_RelatedQuestion.png
│   ├── Related Question Analysis.ipynb
│   ├── Related Question.ipynb
│   └── Topic Model.ipynb
├── data
│   └── emptydata.xlsx
├── model
│   └── lstm_model
│       ├── config.json
│       └── log.json
├── README.md
├── requirements.txt
├── src
│   ├── callbacks.py
│   ├── metric.py
│   ├── modules
│   │   └── net.py
│   ├── mypredictor.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
└── tensorboard
    └── lstm_model
```