from abc import ABC

import torch
import torch.nn as nn

from fengshen import LongformerModel

import numpy as np


class Model(nn.Module, ABC):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'BertClassifier'
        self.word_embedding = LongformerModel.from_pretrained(opt.bert_path)
        for param in self.word_embedding.parameters():
            param.requires_grad = True

        self.embedding_dim = opt.embedding_dim * 1

        self.dropout = nn.Dropout(opt.dropout_rate)

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, opt.hidden_size),
            nn.ReLU(),
            nn.Linear(opt.hidden_size, opt.class_num)
        )
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.classifier[0].weight)
        nn.init.xavier_uniform_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[0].bias)
        nn.init.zeros_(self.classifier[-1].bias)

    def forward(self, X, masks):
        batch_size,seq_len = masks.size()
        global_attention_mask = np.zeros((batch_size,seq_len))
        global_attention_mask[:,0] = 1
        global_attention_mask = torch.tensor(global_attention_mask,device=X.device)
        embed = self.word_embedding(X, attention_mask=masks,global_attention_mask=global_attention_mask).pooler_output

        pooled = self.dropout(embed)

        out = self.classifier(pooled)

        return out


