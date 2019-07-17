import argparse
import logging
import pickle
import modules.net as model_arch
import torch.nn as nn
import json
from preprocess import Embedding, CSDataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import pdb
import os

import torch
from callbacks import ModelCheckpoint, MetricsLogger

def main(args, config_path):
    with open(config_path) as f:
        config = json.load(f)

    embedding_pkl_path = os.path.join(args.model_dir, config["embedding_pkl_path"])
    train_pkl_path = os.path.join(args.model_dir, config["train_pkl_path"])
    val_pkl_path = os.path.join(args.model_dir, config["val_pkl_path"])
    labelEncoder_path = os.path.join(args.model_dir, config["labelEncoder_path"])
    with open(embedding_pkl_path, "rb") as f:
        config["model_parameters"]["embedding"] = pickle.load(f).vectors
        logging.info( "Load embedding from {}".format(embedding_pkl_path))
    with open(train_pkl_path, "rb") as f:
        train = pickle.load(f)
        logging.info( "Load train from {}".format(train_pkl_path))
    with open(val_pkl_path, "rb") as f:
        config["model_parameters"]["valid"] = pickle.load(f)
        logging.info( "Load val from {}".format(val_pkl_path))
    with open(labelEncoder_path, "rb") as f:
        config["model_parameters"]["labelEncoder"] = pickle.load(f)
        logging.info( "Load labelEncoder from {}".format(labelEncoder_path))


    predictor = Predictor(metric=Metric(), **config["model_parameters"])

    if args.load is not None:
        predictor.load(args.load)

    model_checkpoint = ModelCheckpoint(
            os.path.join(args.model_dir, 'model.pkl'),
            'loss', 1, 'all')
    metrics_logger = MetricsLogger(
            os.path.join(args.model_dir, 'log.json'))

    predictor.fit_dataset(train, train.collate_fn, [model_checkpoint, metrics_logger])


class Metric():
    def __init__(self):
        self.n = 0
        self.n_correct = 0

    def update(self, output, gt):
        maxindex = torch.argmax(output, dim=1)
        #pdb.set_trace()
        for i in range(len(gt)):
            self.n += 1
            if gt[i][maxindex[i]] == 1:
                self.n_correct += 1

    def reset(self):
        self.n = 0
        self.n_correct = 0

    def get_score(self):
        score = self.n_correct/self.n
        return "{:.3f}".format(score)
        pass

class Predictor():
    def __init__(self, batch_size=64, max_epochs=100, valid=None, labelEncoder=None, device=None, metric=None,
            learning_rate=1e-3, max_iters_in_epoch=1e20, grad_accumulate_steps=1,
            embedding=None, loss="BCELoss", arch="rnn_net", **kwargs):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.valid = valid
        self.metric = metric
        self.learning_rate = learning_rate
        self.max_iters_in_epoch = max_iters_in_epoch
        self.grad_accumulate_steps = grad_accumulate_steps
        self.le = labelEncoder
        self.num_classes = len(self.le.classes_)

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.epoch = 0

        self.model = getattr(model_arch,arch)(embedding.size(1), self.num_classes, **kwargs)
        print(self.model)
        logging.info("Embedding size: ({},{})".format(embedding.size(0),embedding.size(1)))
        self.embedding = nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)


        self.model = self.model.to(self.device)
        self.embedding = self.embedding.to(self.device)

        logging.info("Learning_rate: {}".format(learning_rate))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.loss={'BCEWithLogitsLoss': nn.BCEWithLogitsLoss()}[loss]
        logging.info("Loss: {}".format(self.loss))

    def fit_dataset(self, data, collate_fn=default_collate, callbacks=[]):
        while self.epoch < self.max_epochs:
            logging.debug("training {}".format(self.epoch))
            dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True,
                    num_workers=10, collate_fn=collate_fn)
            log_train = self._run_epoch(dataloader, True)

            if self.valid is not None:
                logging.debug("evaluating {}".format(self.epoch))
                dataloader = torch.utils.data.DataLoader(self.valid, batch_size=self.batch_size, shuffle=False,
                        num_workers=10, collate_fn=collate_fn)
                log_valid = self._run_epoch(dataloader, False)
            else:
                log_valid = None

            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)

            self.epoch += 1

    def _run_epoch(self, dataloader, training):
        self.model.train(training)

        loss = 0

        if training:
            iter_in_epoch = min(len(dataloader), self.max_iters_in_epoch)
            description = "training"
        else:
            iter_in_epoch = len(dataloader)
            description = "evaluating"

        trange = tqdm(enumerate(dataloader), total=iter_in_epoch, desc=description, ncols=70)
        for i, batch in trange:
            if training:
                if i >= iter_in_epoch:
                    break
                output, batch_loss = self._run_iter(batch, training)
                batch_loss /= self.grad_accumulate_steps

                if i % self.grad_accumulate_steps == 0:
                    self.optimizer.zero_grad()
                batch_loss.backward()
                if (i + 1) % self.grad_accumulate_steps == 0:
                    self.optimizer.step() # update gradient
            else:
                with torch.no_grad():
                    output, batch_loss = self._run_iter(batch, training)

            loss += batch_loss.item()
            self.metric.update(output, batch['labels'])
            trange.set_postfix(loss=loss / (i + 1), **{"Accuracy": self.metric.get_score()})
        loss /= iter_in_epoch
        score = self.metric.get_score()

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        epoch_log["Accuracy"] = score
        print("loss={}".format(loss))
        print("Accuracy={}".format(score))
        return epoch_log

    
    def _run_iter(self, batch, training):
        x = batch['sentences']
        y = batch['labels']
        with torch.no_grad():
            sentence = self.embedding(x.to(self.device))

        logits = self.model.forward(sentence.to(self.device))
        loss = self.loss(logits, y.float().to(self.device))
        return logits, loss

    def _predict_batch(self, x, training):
        sentence = self.embedding(x.to(self.device))
        logits = self.model.forward(sentence.to(self.device))
        return logits
    
    def predict_dataset(self, data, collate_fn=default_collate, batch_size=None, predict_fn=None): # for prediction
        if batch_size is None:
            batch_size = self.batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        self.model.eval()
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False,
                num_workers=10, collate_fn=collate_fn)

        ans = []
        with torch.no_grad():
            trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="predicting", ncols=70)
            for batch in trange:
                batch_y = predict_fn(batch)
                ans.append(batch_y)
        ans = torch.cat(ans, 0)
        return ans
    
    def save(self, path):
            torch.save({
                'epoch': self.epoch + 1,
                'model': self.model.state_dict(),                # A state_dict is simply a Python dictionary object 
                'optimizer': self.optimizer.state_dict()         # that maps each layer to its parameter tensor
                },path)

    def load(self, path):
        saved = torch.load(path)
        self.epoch = saved['epoch']
        self.model.load_state_dict(saved['model'])
        self.optimizer.load_state_dict(saved['optimizer'])




def _parse_args():
    parser = argparse.ArgumentParser(description="Training model.")
    parser.add_argument('model_dir', type=str, help="[input] Path to the model directory.") 
    parser.add_argument('--load', default=None, type=str)
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
