import argparse
import logging
import pickle
import modules.rnn_net as model_arch
import torch.nn as nn

class Predictor():
    def __init(self, batch_size=64, max_epochs=100, valid=None, device=None, metrics={},
            learning_rate=1e-3, max_iters_in_epoch=1e20, grad_accumulate_steps=1,
            embedding=None, dropout_rate=0.2, loss="BCELoss", similarity="inner_product",
            arch="rnn_net", **kwargs):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.valid = valid
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.max_iters_in_epoch = max_iters_in_epoch
        self.grad_accumulate_steps = grad_accumulate_steps

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.epoch = 0


        self.model = getattr(model_arch,arch)(embedding.size(1), similarity=similarity)
        print(self.model)
        self.embedding = nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)

        logging.info("Learning_rate: {}".format(learning_rate))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.loss={'BCEWithLogitsLoss': nn.BCEWithLogitsLoss()}[loss]
        logging.info("Loss: {}".format(self.loss))

    def fit_dataset(self, data, collate_fn=default_collate, callbacks=[]):
        while self.epoch < self.max_epochs:
            dataloader = torch.tuils.data.DataLoader(data, batch_size=self.batch_size, shuffle=True,
                    num_worker=10,collate_fn=collate_fn)
            log_train = self._run_epoch(dataloader, True)
            if self.valid is not None:
            dataloader = torch.tuils.data.DataLoader(self.valid, batch_size=self.batch_size, shuffle=False,
                    num_worker=10,collate_fn=collate_fn)
            log_valid = self._run_epoch(dataloader, False)

            self.epoch += 1
    def _run_epoch(self, dataloader, training):
        self.model.train(training)

        loss = 0


        if training:
            iter_in_epoch = min(len(dataloader), self max_iters_in_epoch)
            description = "training"
        else:
            iter_in_epoch = min(len(dataloader))
            description = "evaluating"

        trange = tqdm(enumerate(dataloader), total=iter_in_epoch, desc = description)
        for i, batch in trange:
            if training:
                if i >= iter_in_epoch:
                    break
                output, batch_loss = self._run_iter(batch, training)
                batch_loss /= self.grad_accmulate_steps

                if i % self.grad_accumulate_steps == 0:
                    self.optimizer.zero_grad()
                batch_loss.backward()
                if (i + 1) % self.grad_accumulate_steps == 0:
                    self.optimizer.step() # update gradient
            else:
                with torch.no_grad():
                    output, batch_loss = self._run_iter(batch, training)

            loss += batch_loss.item()
        loss /= iter_in_epoch
        print("loss={}\n".format(loss))
        return 

    def _run_iter(self, batch, training):
        x = batch[:,:-2]
        y = batch[:,-1]
        with torch.no_grad():
            sentence = self.embedding(x.to(self.device))
        logits = self.model.forward(sentence.to(self.device))
        loss = self.loss(logits,y.float().to(self.device))
        return logits, loss

    def _predict_batch(self, x, training):
        sentence = self.embedding(x.to(self.device))
        logits = self.model.forward(sentence.to(self.device))
        return logits
    
    def predict_dataset(self, data, collate_fn=default_collate, batch_size=None, predict_fn=None):
        if batch_size is None:
            batch_size = self.batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        self.model.eval()
        dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False,
                num_workers=4, collate_fn=collate_fn)

        ans = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
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



def main(args):
    with open(args.config_path) as f:
        config = json.load(f)

    logging.info( "Load embedding from {}".format(config["embedding_pkl_path"]))
    with open(config["embedding_pkl_path"]) as f:
        embedding = pickle.load(f)
    with open(self.config["train_x_pkl_path"]) as f:
        train_x = pickle.load(f)
        logging.info( "Load train_x from {}".format(self.config["train_x_pkl_path"]))
    with open(self.config["train_y_pkl_path"]) as f:
        train_y = pickle.load(f)
        logging.info( "Load train_y from {}".format(self.config["train_y_pkl_path"]))
    with open(self.config["val_x_pkl_path"]) as f:
        val_x = pickle.load(,f)
        logging.info( "Load val_x from {}".format(self.config["val_x_pkl_path"]))
    with open(self.config["val_y_pkl_path"]) as f:
        val_y = pickle.load(f)
        logging.info( "Load val_y from {}".format(self.config["val_y_pkl_path"]))

    predictor = Predictor()





def _parse_args():
    parser = argparse.ArgumentParser(description="Training model.")
    parser.add_argument('config_path', type=str, default="./config.json", help="[input] Path to the config file.") 
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    args = _parse_args()
    logging.info('Loading congiguration file from {}'.format(args.config_path))
    main(args)
