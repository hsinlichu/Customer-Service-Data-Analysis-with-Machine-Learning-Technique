import argparse
import logging
import pickle
import torch.nn as nn
import json
from preprocess import Embedding, CSDataset
import pdb
import os

import torch
from mypredictor import Predictor
from callbacks import ModelCheckpoint, MetricsLogger, Tensorboard
from metric import Metric

def main(args, config_path):
    logging.info('Loading configuration file from {}'.format(config_path))
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

    tensorboard = Tensorboard(config["tensorboard"])

    logging.info("start training!")
    predictor.fit_dataset(train, train.collate_fn, [model_checkpoint, metrics_logger, tensorboard])



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
