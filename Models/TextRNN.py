from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np

class Model(nn.Module, ABC):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'TextRNN'
        self.word_embedding = nn.Embedding(21128, opt.embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.rnn = nn.LSTM(opt.embedding_dim, opt.embedding_dim, 2,bidirectional=False, batch_first=True)
        self.classifier = nn.Linear(opt.embedding_dim,opt.class_num)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, X,masks):
        batch_size = X.size(0)
        embed = self.word_embedding(X)

        embed = self.dropout(embed)

        rnn_out, _ = self.rnn(embed)
        out = self.classifier(rnn_out[np.arange(batch_size), masks.sum(dim=1)-1, :])  # 句子最后时刻的 hidden state
        return out
