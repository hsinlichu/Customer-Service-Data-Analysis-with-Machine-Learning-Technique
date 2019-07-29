import torch
import torch.nn as nn
import pdb
import logging

class rnn_net(nn.Module):
    def __init__(self, dim_embeddings, num_classes, similarity="inner_product", hidden_size=128,
            num_layers=1,  rnn_dropout=0.2, clf_dropout=0.3, bidirectional=False):
        super(rnn_net, self).__init__()
        self.hidden_size = hidden_size
        self.dim_embeddings = dim_embeddings
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bool(bidirectional)
        self.clf_dropout = clf_dropout
        self.flag = 1

        self.rnn = nn.LSTM(input_size=self.dim_embeddings, hidden_size=self.hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True) # , dropout=rnn_dropout
        self.clf = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.clf_dropout),
                nn.Linear(hidden_size // 2, num_classes)
                )

    def forward(self, sentence):
        # sentence: torch.Size([64, 80, 300])
        sentence_out, hidden = self.rnn(sentence)
        if not self.flag:
            print(sentence_out.size(),hidden.size()) # torch.Size([32, 80, 128]) torch.Size([1, 32, 128])
            self.flag = 1
        #pdb.set_trace() # breakpoint
        tmp = sentence_out[:,-1,:]

        score = self.clf(tmp)
        #print(score.size())) # torch.Size([64, 37])
        return score
