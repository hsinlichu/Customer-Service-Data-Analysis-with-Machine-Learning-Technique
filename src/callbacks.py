import json

class Callback:
    def __init__():
        pass
    def on_epoch_end(log_train, log_valid, model):
        pass

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
