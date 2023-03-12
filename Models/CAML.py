from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Model(nn.Module, ABC):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'CAML'
        self.word_embedding = nn.Embedding(21128, opt.embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.conv1d = nn.Conv1d(opt.embedding_dim,opt.embedding_dim,kernel_size=3,padding = 1)
        
        self.U = nn.Linear(opt.embedding_dim, opt.class_num)

        self.classifier = nn.Linear(opt.embedding_dim,opt.class_num)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        nn.init.xavier_uniform_(self.U.weight)
        nn.init.zeros_(self.U.bias)

    def forward(self, X):
        embed = self.word_embedding(X)

        embed = self.dropout(embed)

        conv_out = self.conv1d(embed.transpose(1,2)) # [batch_size,768,seq_len]

        conv_out = torch.tanh(conv_out) 

        alpha = torch.softmax(self.U.weight @ conv_out,dim=2) # [batch_size,class_num,seq_len]

        attn_out = alpha @ conv_out.transpose(1,2) # [batch_size, class_num, hidden_size]

        out = (attn_out * self.classifier.weight.unsqueeze(0)).sum(dim=2) + self.classifier.bias.unsqueeze(0)

        return out


