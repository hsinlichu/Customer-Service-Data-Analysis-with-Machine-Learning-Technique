import math
import pandas as pd
import logging
import numpy as np
import argparse
import json
from tqdm import tqdm
import nltk
import re
import pickle
import os

from sklearn import preprocessing
import torch
from torch.utils.data import Dataset


def main(args, config_path):
    logging.info('Loading congiguration file from {}'.format(config_path))
    with open(config_path) as f:
        config = json.load(f)

    embedding = Embedding(config["embedding_path"])
    embedding_path = os.path.join(model_dir, self.config["embedding_pkl_path"])
    with open(embedding_path, "wb") as f:
        pickle.dump(embedding, f)
    logging.info( "Save embedding to {}".format(embedding_path))
    preprocessor = Preprocessor(config, embedding, args.model_dir)



class CSDataset(Dataset):
    def __init__(self, data, padding, num_classes, shuffle=False,  padded_len=80):
        self.data = data
        self.padded_len = padded_len
        self.shuffle = shuffle
        self.padding = padding
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        batch = {}
        batch['labels'] = torch.tensor(self._to_one_hot([r[-1] for r in datas]))
        batch['sentences'] = torch.tensor([ self._pad_to_len(r[:-2]) for r in datas])
        return batch

    def _to_one_hot(self, y):
        logging.info("Number of Classes: {}".format(self.num_classes))
        matrix = torch.eye(self.num_classes)
        ret = [matrix[r].tolist() for r in y]
        return ret

    def _pad_to_len(self,arr):
        if len(arr) > self.padded_len:
            arr = arr[:self.padded_len]
        while len(arr) < self.padded_len:
            arr.append(self.padding)
        return arr


class Embedding:
    def __init__(self, embedding_path, seed=1357):
        self.word_dict = {}
        self.vectors = []
        torch.manual_seed(seed)
        self.load_embedding(embedding_path)

    
    def load_embedding(self, embedding_path):
        with open(embedding_path, encoding="utf-8") as f:
            for i, line in enumerate(tqdm(f)):
                if i == 0:          # skip header
                    continue
                #if i == 1000:          # skip header
                #    break

                row = line.rstrip().split(' ') # rstrip() method removes any trailing characters (default space)
                word, vector = row[0], row[1:]
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
        if vector is None:
            vector = torch.empty(1,self.get_dim())
            torch.nn.init.uniform_(vector)
        vector.view(1,-1)
        self.word_dict[word] = len(self.word_dict)
        self.vectors = torch.cat((self.vectors, vector), 0)

    def to_index(self, word):
        word = word.lower()
        if word in self.word_dict:
            return self.word_dict[word] 
        else:
            return self.word_dict['<unk>']

class Preprocessor:
    def __init__(self, config, embedding, model_dir):
        nltk.download('punkt')
        #self.logging = logging.getLogger(name=__name__)
        self.config = config
        self.le = preprocessing.LabelEncoder()
        self.num_classes = None
        self.data = None
        self.processed = []
        self.embedding = embedding 
        self.get_dataset()
        self.split_data()
        self.export(model_dir)

    def get_dataset(self, n_workers=10):
        logging.info('Getting Dataset...')
        self.read_data()
        cnt = 0
        total = 0

        for x,y in tqdm(self.data, total=len(self.data), desc="Processing", ascii=True): 
            x = re.sub('<[^<]*?/?>', ' ', x)        # remove all html tag
            x = re.sub('https?:\/\/[^ ]*', ' ', x)  # remove all url
            x = re.sub('\S*@\S*\s?', ' ', x)        # remove all email address
            x = re.sub('[^a-z A-Z]', ' ', x)        # remove all non-english alphabat
            self.processed.append(self.sentence_to_indices(x) + [y])
            cnt += 1 
            total += len(self.processed[-1]) - 1
        logging.info("Average Sentence length: {}".format(total / cnt))
            
            
    def read_data(self):
        data_path = self.config['data_path']
        logging.info("Pandas read {}".format(data_path))
        df = pd.read_excel(data_path)
        df = df.dropna() # drop nan entry
        # df[pd.isnull(df).any(axis=1)]

        self.le.fit(df['catName'].unique())
        #print(self.le.transform(df.loc[:,'catName']))
        df.loc[:,'catName'] = self.le.transform(df.loc[:,'catName'])
        self.num_classes = len(self.le.classes_)
        logging.info("Number of classes: {}".format(self.num_classes))
        self.data = df[['question','catName']].to_numpy() # convert to numpy array
        logging.info("Dataset shape {}".format(self.data.shape))

    def split_data(self):
        portion = self.config["training_portion"]
        logging.info("Spliting Dataset with portion: {}".format(portion))
        np.random.shuffle(self.processed)
        cut = math.ceil(len(self.data) * portion)
        self.train = self.processed[:cut]
        self.val = self.processed[cut:]
        logging.info("Training set shape: {} | Validation set shape: {}".format(len(self.train), len(self.val)))

    def export(self, model_dir):
        padding = self.embedding.to_index('<pad>')
        train = CSDataset(self.train, padding, self.num_classes)
        val = CSDataset(self.val, padding, self.num_classes)

        le_path = os.path.join(model_dir, self.config["labelEncoder_path"])
        train_path = os.path.join(model_dir, self.config["train_pkl_path"])
        val_path = os.path.join(model_dir, self.config["val_pkl_path"])

        with open(le_path, "wb") as f:
            pickle.dump(self.le,f)
            logging.info( "Save labelEncoder to {}".format(le_path))
        with open(train_path, "wb") as f:
            pickle.dump(train,f)
            logging.info( "Save train to {}".format(train_path))
        with open(val_path, "wb") as f:
            pickle.dump(val,f)
            logging.info( "Save val to {}".format(val_path))



    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        return tokens

    def sentence_to_indices(self, sentence):
        sentence = self.tokenize(sentence)

        ret = []
        for word in sentence:
            ret.append(self.embedding.to_index(word))
        return ret

def _parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing and training.")
    parser.add_argument('model_dir', type=str, help="[input] Path to the model directory.") 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    args = _parse_args()
    config_path = os.path.join(args.model_dir, "config.json")

    if os.path.isfile(config_path):
        main(args, config_path)
    else:
        logging.error("config.json NOT exist in {}.".format(config_path))
        exit(1)
    
