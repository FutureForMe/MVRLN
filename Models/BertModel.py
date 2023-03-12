from abc import ABC

import torch
import torch.nn as nn

from transformers import AutoModel


class Model(nn.Module, ABC):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'BertClassifier'
        self.word_embedding = AutoModel.from_pretrained(opt.bert_path)
        for param in self.word_embedding.parameters():
            param.requires_grad = True

        self.embedding_dim = opt.embedding_dim * 3

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
        chief_embed = self.word_embedding(X[0], attention_mask=masks[0]).pooler_output
        now_embed = self.word_embedding(X[1], attention_mask=masks[1]).pooler_output
        past_embed = self.word_embedding(X[2], attention_mask=masks[2]).pooler_output

        pooled = torch.cat((chief_embed, now_embed, past_embed), dim=1)

        pooled = self.dropout(pooled)

        out = self.classifier(pooled)

        return out


