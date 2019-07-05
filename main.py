import torch
import math
import pandas as pd
import logging
from sklearn import preprocessing
import numpy as np
import argparse
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import nltk

class CSDataset(Dataset):
    def __init__(self, data, padded_len=300):
        self.data = data
        self.padded_len = padded_len
        self.shuffle = shuffle

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


class Embedding:
    def __init__(self, embedding_path, seed=1357):
        self.word_dict = {}
        self.vectors = []
        torch.manual_seed(seed)
        self.load_embedding(embedding_path)

    
    def load_embedding(self, embedding_path):
        with open(embedding_path, encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f)):
                if i == 0:
                    continue
                if i == 10000:
                    break
                row = line.rstrip().split(' ')
                word, vector  = row[0], row[1:]
                word = word.lower()
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)
                    vector = [float(n) for n in vector] 
                    self.vectors.append(vector)
        self.vectors = torch.tensor(self.vectors)

        if '<unk>' not in self.word_dict:
            self.add('<unk>')
        if '<pad>' not in self.word_dict:
            self.add('<pad>', torch.zeros(1, self.get_dim()))

        logging.info("Embedding size: {}".format(self.vectors.size()))

    def get_dim(self):
        return len(self.vectors[0])

    def add(self, word, vector=None):
        print(word,vector)
        if vector is None:
            vector = torch.empty(1,self.get_dim())
            torch.nn.init.uniform_(vector)
        vector.view(1,-1)
        self.word_dict[word] = len(self.word_dict)
        print(self.vectors.shape, vector.shape)
        self.vectors = torch.cat((self.vectors, vector), 0)

    def to_index(self, word):
        word = word.lower()
        if word in self.word_dict:
            return self.word_dict[word] 
        else:
            return self.word_dict['<unk>']

class Preprocessor:
    def __init__(self, config, embedding):
        #self.logging = logging.getLogger(name=__name__)
        self.config = config
        self.le = preprocessing.LabelEncoder()
        self.data = None
        self.embedding = embedding 
        self.get_dataset()
        #self.split_data()

    def get_dataset(self, n_workers=4):
        logging.info('Getting Dataset...')
        self.read_data()

        processed = []
        for x,y in tqdm(self.data):
            self.sentence_to_indices(x)
            
            




    def read_data(self):
        data_path = config['data_path']
        logging.info("Pandas read {}".format(data_path))
        df = pd.read_excel(data_path)
        df = df.dropna() # drop nan entry
        # df[pd.isnull(df).any(axis=1)]

        self.le.fit(df['catName'].unique())
        #print(self.le.transform(df.loc[:,'catName']))
        df.loc[:,'catName'] = self.le.transform(df.loc[:,'catName'])
        logging.info("Number of classes: {}".format(len(self.le.classes_)))
        self.data = df[['question','catName']].to_numpy() # convert to numpy array
        logging.info("Dataset shape {}".format(self.data.shape))

    def split_data(self):
        portion = self.config["training_portion"]
        logging.info("Spliting Dataset with portion: {}".format(portion))
        np.random.shuffle(self.data)
        cut = math.ceil(len(self.data) * portion)
        self.train_x, self.train_y = self.data[:cut,0],self.data[:cut,1]
        self.val_x, self.val_y = self.data[cut:,0],self.data[cut:,1]
        logging.info("Training set len: {} | Validation set len: {}".format(self.train_x.shape[0], self.val_x.shape[0]))

    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def sentence_to_indices(self, sentence):
        sentence = self.tokenize(sentence)
        print(sentence)
        

def _parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing and training.")
    parser.add_argument('config_path', type=str, default="./config.json", help="[input] Path to the config file.") 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    args = _parse_args()
    logging.info('Loading congiguration file from {}'.format(args.config_path))
    with open(args.config_path) as f:
        config = json.load(f)


    embedding = Embedding(config["embedding_path"])
    preprocessor = Preprocessor(config, embedding)





