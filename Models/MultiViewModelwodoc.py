from abc import ABC

import torch
import torch.nn as nn
from transformers import AutoModel
from fengshen import LongformerModel
# from transformers import RoFormerModel as BertModel


import torch.nn.functional as F
import numpy as np

from Module.GCN_layers import GraphConvolution
# from Module.Self_Attention import MultiHeadAttention
from utils import normalize, normalize_adj, normalize_features


class Model(nn.Module, ABC):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.model_name = 'BertCoGAttV2'
        self.class_num = opt.class_num
        self.embedding_dim = opt.embedding_dim

        self.use_longformer = 'Longformer' in opt.bert_path
        if not self.use_longformer:
            self.word_embedding = AutoModel.from_pretrained(opt.bert_path)
        else:
            self.word_embedding = LongformerModel.from_pretrained(opt.bert_path)
        for param in self.word_embedding.parameters():
            param.requires_grad = True

        # 模型第一路(2)双向LSTM
        self.bi_lstm = nn.LSTM(768, 768//2, 1,bidirectional=True, batch_first=True)

        # 模型第一路(3)标签注意力
        self.label_attn = nn.Linear(self.embedding_dim,self.class_num) # [self.class_num.self.embedding_dim]

        # 模型第二路(1)实体嵌入[bert预训练生成]
        embedding = torch.tensor(opt.entities_pretrain_embed.astype(np.float32))
        self.entity_embed = nn.Embedding.from_pretrained(embedding)
        # self.entity_embed = nn.Embedding(opt.entity_size,opt.embedding_dim)
        self.dropout = nn.Dropout(opt.dropout_rate)
        self.dropout2 = nn.Dropout(opt.dropout_rate)

        # GCN融合标签共现关系
        # self.graph_embedding = nn.Embedding(self.graph_nodes_num, self.embedding_dim)
        self.gc1 = GraphConvolution(2*self.embedding_dim, 2*self.embedding_dim)
        self.gc2 = GraphConvolution(2*self.embedding_dim, 2*self.embedding_dim)

        # 分类平面
        self.classifier = nn.Linear(2*self.embedding_dim, self.class_num)

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_uniform_(self.label_attn.weight)
        nn.init.zeros_(self.label_attn.bias)

        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)


    def forward(self, X, masks,entities,label_matrix,ent_label_matrix):


        # 字符级别&&文档级别
        char_level_embed=[]
        doc_level_embed=[]
        mask_tmp = []
        for x,mask in zip(X,masks):
            if not self.use_longformer:
                last_hidden_state = self.word_embedding(x,mask).last_hidden_state
            else:
                batch_size,seq_len = mask.size()
                global_attention_mask = np.zeros((batch_size,seq_len))
                global_attention_mask[:,0] = 1
                global_attention_mask = torch.tensor(global_attention_mask,device=x.device)
                last_hidden_state = self.word_embedding(x,mask,global_attention_mask = global_attention_mask).last_hidden_state

            char_level_embed.append(last_hidden_state[:,1:])
            doc_level_embed.append(last_hidden_state[:,0])
            mask_tmp.append(mask[:,1:])
        char_level_embed=torch.cat(char_level_embed,dim=1)
        char_level_embed=self.dropout(char_level_embed)

        masks = torch.cat(mask_tmp,dim=1)
        all_masks = - (1 - masks.unsqueeze(1)) * 1e7
        
        # 文本注意力
        pooled,(_,_) = self.bi_lstm(char_level_embed)         # [batch_size,seq_len,2*hidden_size]

        alpha = torch.softmax(self.label_attn.weight @ pooled.transpose(1,2) + all_masks, dim = 2) # [batch_size, num_labels, seq_len]
        char_level_label_embed = alpha @ pooled # [batch_size, num_labels, 768]

        # 实体级别
        entity_level_embed = self.entity_embed.weight.unsqueeze(0) * entities.unsqueeze(2) # [batch_size,entity_num,768]
        entity_level_embed = self.dropout2(entity_level_embed)
        entity_level_label_embed = ent_label_matrix.T @ entity_level_embed # [batch_size, num_labels, 768]



        # 文档级别
        doc_level_embed = torch.stack(doc_level_embed,dim=1).sum(dim=1)
        doc_level_embed = doc_level_embed.unsqueeze(1).repeat(1,self.class_num,1)

        fuse_features = torch.cat((char_level_label_embed,entity_level_label_embed),dim=2)

        # 经过两层gcn
        label_matrix = normalize_adj(label_matrix)
        fuse_features1 = torch.relu(self.gc1(fuse_features,label_matrix)) # [batch_size, num_labels, 768]
        fuse_features2 = torch.relu(self.gc2(fuse_features1,label_matrix))

        out = (fuse_features + fuse_features1+fuse_features2) * self.classifier.weight.unsqueeze(0) # 残差
        out = out.sum(dim = 2) # [batch_size, num_labels]

        return out

