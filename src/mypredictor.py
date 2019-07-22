import logging
import modules.net as model_arch
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import torch
import torch.nn as nn


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

        self.metric.reset()


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

    def _predict_batch(self, x):
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
        solutions = []
        with torch.no_grad():
            trange = tqdm(enumerate(dataloader), total=len(dataloader), desc="predicting", ncols=70)
            for i, batch in trange:
                x = batch["sentences"]
                solution = batch["labels"]
                batch_y = predict_fn(x) #batch
                solutions.append(solution)
                ans.append(batch_y)
        ans = torch.cat(ans, 0)
        solutions = torch.cat(solutions, 0)
        return ans, solutions
    
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


