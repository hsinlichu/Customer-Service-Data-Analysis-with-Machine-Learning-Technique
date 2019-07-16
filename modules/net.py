import torch
import torch.nn as nn

class rnn_net(nn.module):
    def __init__(self, dim_embeddings, similarity="inner_product", hidden_size=128,
            num_layers=1, num_classes=64, rnn_dropout=0.2, clf_dropout=0.3, bidirectional=False):
        self.hidden_size = hidden_size
        self.dim_embeddings = dim_embeddings
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(dim_embeddings, hidden_size, num_layers, rnn_dropout, bidirectional, batch_first=True)
        self.clf = nn.Sequential(
                nn.Linear(hidden_size,hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.dropout(self.clf_dropout),
                nn.Linear(num_classes)
                )

    def forward(self, sentence):
        sentence_out = self.rnn(sentence)
        score = self.clf(sentence_out)
