from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'TextRCNN'

        self.word_embedding = nn.Embedding(21128,opt.embedding_dim,padding_idx=0)

        self.dropout = nn.Dropout(opt.dropout_rate)

        self.rnn = nn.LSTM(opt.embedding_dim, opt.embedding_dim//2, 2,bidirectional=True, batch_first=True)

        self.classifier = nn.Linear(2*opt.embedding_dim,opt.class_num)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, X):
        embed = self.word_embedding(X)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        embed = self.dropout(embed)
        out, _ = self.rnn(embed)
        out = torch.cat((embed, out), 2) # [batch_size,seq_len,2*embed_dim]
        out,_ = torch.max(out,dim=1)
        out = self.classifier(out)
        return out