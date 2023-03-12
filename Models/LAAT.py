from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Model(nn.Module):
    def __init__(self,opt):
        super(Model, self).__init__()
        self.word_embedding = nn.Embedding(21128, opt.embedding_dim, padding_idx=0)

        self.dropout = nn.Dropout(opt.dropout_rate)
        self.bi_lstm = nn.LSTM(opt.embedding_dim, opt.embedding_dim//2, 2,
                            bidirectional=True, batch_first=True)

        self.w = nn.Parameter(torch.FloatTensor(opt.embedding_dim,768))
        nn.init.xavier_normal_(self.w)
        self.u = nn.Parameter(torch.FloatTensor(768,opt.class_num))
        nn.init.xavier_normal_(self.u)

        self.final1 = nn.Linear(opt.embedding_dim, 1024)
        nn.init.xavier_normal_(self.final1.weight)
        self.final2 = nn.Linear(1024, opt.class_num)
        nn.init.xavier_normal_(self.final2.weight)

    def forward(self, x):    

        embed = self.word_embedding(x)             # [batch_size,seq_len,embeding_size]
        embed = self.dropout(embed)              # dropout
        # bi-lstm
        H,(_,_) = self.bi_lstm(embed)         # [batch_size,seq_len,2*hidden_size]
        
        # label attn layer
        Z = torch.tanh(H @ self.w)            # [batch_size,seq_len,attn_d]
        A = torch.softmax(Z @ self.u,dim = 1) # [batch_size,seq_len,labels_num]
        V = A.transpose(1,2) @ H              # [batch_size,labels_num,2*hidden_size]

        # output layer
        V = torch.relu(self.final1(V))        # [batch_size,labels_num,ffn_size]
        y_hat = self.final2.weight.mul(V).sum(dim=2).add(self.final2.bias) # [batch_size,labels_num]


        return y_hat