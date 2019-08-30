# Customer Service Data Analysis with Machine Learning Technique 
This project was done when I interned at [Cyberlink](https://www.cyberlink.com/index_en_US.html). I was responsible for analyzing with machine learning technique. 

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

Note: Related analysis result pictures are stored in [analysis/picture/](https://github.com/james60708/Cyberlink-Intern/tree/master/analysis/picture) folder.


## Dataset
This customer service data is around 0.1 million customer feedback sentence with subject, question sentence, and user selected question type.

![](https://i.imgur.com/zwK5NUf.png)




## Supervised text classification
I use simply GRU model to classify text sentence into their group. The input is question string sentence and the output is its class. The ground truth is user selected question type.

### How to run
* Preprocessing: `python ./src/preprocess.py <directory/which/contain/model/config.json> [-e ./path/to/embedding.pkl]`
    
    The last argument `-e` is optional. Because processing `embedding.pkl` take long time, you can just use this argument and specify your `embedding.pkl` to save your time.  
    
    Ex: `python ./src/preprocess.py ./model/lstm_model -e ./model/lstm_model/embedding.pkl`
* Training: `python ./src/train.py <directory/which/contain/model/config.json>`

    Ex: `python ./src/train.py ./model/lstm_model`
* Run tensorboard: `tensorboard --logdir tensorboard`

### Model
![](https://i.imgur.com/Nr2ZTDW.png)

## Unsupervised text clustering
To get more insight into this customer service dataset, I use unsupervised text clustering to see whether it can discover something interesting.

In order to cluster text, we need to build vector to represent each sentence first. Therefore, I experiment two sentence representation method.

After we turn each sentence into vector, I use Kmeans and DBSCAN to cluster those data, and compare those results.



### Sentence representation
* Doc2Vec
* TF-IDF

### Clustering Algorithm
* Kmeans
* DBSCAN

[Result and code](https://github.com/james60708/Cyberlink-Intern/blob/master/analysis/Customer%20Service%20Data%20Clustering.ipynb)

Note: Below graph are the clustering result using Doc2Vec sentence representation by Kmeans clustering algorithm.
![](https://i.imgur.com/xn2t0Lq.png)

## Topic Modeling - LDA
I also use LDA which is a generative probability model to figure out some latent topics. The result below are visualized by [pyLDAvis].(https://github.com/bmabey/pyLDAvis)
![](https://i.imgur.com/rjk7G3f.png)

## Find related question based on BERT.
Since if we can find related question and its corresponding answer, we can find the most similar and return to customer before the customer submit its question feedback.

In this way, we can solve customer's problem more quickly and also reduce the repeated questions that need human to answer. 

Here, we use BERT, the state-of-the-art NLP model in 2018, to build sentence representation, and simply use cosine similarity to find related question. [Model Code](https://github.com/james60708/Cyberlink-Intern/blob/master/analysis/Related%20Question.ipynb)

Above method is very naive and has lots of space for improvement, so we simply serve it as a baseline to see its potential. [Performance Analysis](https://github.com/james60708/Cyberlink-Intern/blob/master/analysis/Related%20Question%20Analysis.ipynb)


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