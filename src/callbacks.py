import json
import math
from tensorboardX import SummaryWriter
import os



class Callback:
    def __init__():
        pass
    def on_epoch_end(log_train, log_valid, model):
        pass

class Tensorboard(Callback):
    def __init__(self, comment):
        dir_name = "tensorboard"
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        path = os.path.join(dir_name, comment)
        #self.writer = SummaryWriter(path)
        self.writer = SummaryWriter(path)

    def on_epoch_end(self, log_train, log_valid, model):
        self.writer.add_scalar("Train_Accuracy",log_train["Accuracy"], model.epoch)
        self.writer.add_scalar("Train_Loss",log_train["loss"], model.epoch)
        self.writer.add_scalar("Valid_Accuracy",log_valid["Accuracy"], model.epoch)
        self.writer.add_scalar("Valid_Loss",log_valid["loss"], model.epoch)



class MetricsLogger(Callback):
    def __init__(self, log_dest):
        self.history = {
                'train': [],
                'valid': []
                }
        self.log_dest = log_dest

    def on_epoch_end(self, log_train, log_valid, model):
        log_train['epoch'] = model.epoch
        log_valid['epoch'] = model.epoch
        self.history['train'].append(log_train)
        self.history['valid'].append(log_valid)
        with open(self.log_dest, "w") as f:
            json.dump(self.history, f, indent='    ')

class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='loss', verbose=0, mode='min'):
        self._filepath = filepath
        self._monitor = monitor
        self._best = math.inf if mode == 'min' else -math.inf
        self._mode = mode

    def on_epoch_end(self, log_train, log_valid, model):
        score = log_valid[self._monitor]
        if self._mode == 'min':
            if score < self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    logging.info("Best model saved {}".format(score))
        elif self._mode == 'max':
            if score > self._best:
                self._best = score
                model.save(self._filepath)
                if self._verbose > 0:
                    logging.info("Best model saved {}".format(score))
        elif self._mode == 'all':
            model.save("{}.{}".format(self._filepath, model.epoch))
