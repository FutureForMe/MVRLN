from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class Model(nn.Module, ABC):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'TextCNN'
        self.word_embedding = nn.Embedding(21129,opt.embedding_dim,padding_idx=0)

        self.dropout = nn.Dropout(opt.dropout_rate)
        self.num_filters = 256
        
        self.filter_sizes = (3, 4, 5)
        self.convs = nn.ModuleList(
                [nn.Conv1d(opt.embedding_dim, self.num_filters, k) for k in self.filter_sizes])
        self.classifier = nn.Linear(3 * self.num_filters, opt.class_num)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def conv_and_pool(self,embed,conv):
        ans,_ = torch.max(conv(embed.transpose(1,2)),dim=2) #[batch_size,num_filters]
        return ans

    def forward(self, X):
        embed = self.word_embedding(X)
        embed = self.dropout(embed)
        # [batch_size,seq_len,hidden_size]

        cnn_out = torch.cat([self.conv_and_pool(embed, conv) for conv in self.convs], 1)

        out = self.classifier(cnn_out)

        return out
