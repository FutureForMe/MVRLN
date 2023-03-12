"""
   使用pytorch中的Dataloader 
"""
import random

import torch
import json
import torch.utils.data as data
from torch.utils.data import Sampler
import numpy as np
import pickle as pk

import os
from transformers import AutoTokenizer
import copy
import json

from utils import get_age

PAD, CLS, SEP = '[PAD]', '[CLS]', '[SEP]'


class Dataset(data.Dataset):

    def __init__(self, filename, opt):
        super(Dataset, self).__init__()
        self.data_dir = filename  # data_path
        self.label_idx_path = opt.label_idx_path
        self.label_smooth_lambda = opt.label_smooth_lambda
        self.label2id = opt.label2id
        self.entity2id = opt.entity2id
        self.batch_size = opt.batch_size

        self.bert_path = opt.bert_path
        self.class_num = opt.class_num
        self.data = []
        self.label_matrix = opt.label_matrix
        self.ent_label_matrix = opt.ent_label_matrix
        self.label_distribution = np.array([0 for _ in range(opt.class_num)],dtype=np.float64)
        self._preprocess()


    def _preprocess(self):
        print("Loading data file...")
        tokenizer = AutoTokenizer.from_pretrained(self.bert_path)

        with open(self.data_dir, 'r', encoding='UTF-8')as f:
            dicts = json.load(f)

        for dic in dicts:
            if '主诉' not in dic["主诉"]:
                dic['主诉'] = '主诉：'+dic['主诉']
            if '现病史' not in dic["现病史"]:
                dic['现病史'] = '现病史：'+dic['现病史']
            if '既往史' not in dic["既往史"]:
                dic['既往史'] = '既往史：'+dic['既往史']
            chief_complaint = '性别：' + dic['性别'] + '；年龄：'+get_age(dic['年龄']) + '；'+ dic["主诉"]
            now_history, past_history = dic["现病史"], dic["既往史"]

            doc = chief_complaint + '[SEP]' + now_history + '[SEP]' + past_history

            chief_complaint = tokenizer(chief_complaint[:50])
            now_history = tokenizer(now_history[:500])
            past_history = tokenizer(past_history[:300])
            doc = tokenizer(doc[:850])

            item_entities = set(dic['疾病实体']) | set(dic['治疗实体']) | set(dic['检查实体']) | set(dic['症状实体']) | set(dic['检查结果实体'])

            entities = np.array([0 if entity not in item_entities else 1 for entity in self.entity2id])

            label = np.array([self.label_smooth_lambda if label not in dic['出院诊断'] else 1-self.label_smooth_lambda \
                              for label in self.label2id])
            self.label_distribution += label # 

            self.data.append((chief_complaint['input_ids'], now_history['input_ids'], past_history['input_ids'],doc['input_ids'],
                              chief_complaint['attention_mask'], now_history['attention_mask'], past_history['attention_mask'],doc['attention_mask'],
                              entities, self.label_matrix, self.ent_label_matrix, label))
        self.label_distribution /= (np.sum(self.label_distribution) + 1e-3)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chief_idxes, now_idxes, past_idxes,doc_idxes, \
            chief_mask, now_mask, past_mask,doc_mask, \
            entities, label_matrix, ent_label_matrix, label = self.data[idx]
        
        entities = torch.tensor(entities, dtype=torch.float32).unsqueeze(0)
        label_matrix=torch.tensor(label_matrix,dtype=torch.float32)
        ent_label_matrix=torch.tensor(ent_label_matrix,dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)


        return chief_idxes, now_idxes, past_idxes,doc_idxes, \
                chief_mask, now_mask, past_mask,doc_mask, \
                entities, label_matrix, ent_label_matrix, label


def collate_fn(X):
    X = list(zip(*X))
    chief_idxes, now_idxes, past_idxes,doc_idxes, \
    chief_mask, now_mask, past_mask,doc_mask, \
    entities, label_matrixs, ent_label_matrixs, labels = X

    # 最长pad
    idxs = [chief_idxes, now_idxes, past_idxes,doc_idxes]
    masks = [chief_mask, now_mask, past_mask,doc_mask]
    for j,(idx,mask) in enumerate(zip(idxs,masks)):
        max_len = max([len(t) for t in idx])
        for i in range(len(idx)):
            idx[i].extend([0 for _ in range(max_len - len(idx[i]))])  # pad
            mask[i].extend([0 for _ in range(max_len - len(mask[i]))])
        idxs[j] = torch.tensor(idx,dtype = torch.long)
        masks[j] = torch.tensor(mask,dtype = torch.long)

    batch_entities=torch.cat(entities,0)
    batch_labels = torch.cat(labels, 0)

    return idxs[0],idxs[1],idxs[2],idxs[3],masks[0],masks[1],masks[2],masks[3],\
           batch_entities, label_matrixs[0],ent_label_matrixs[0], batch_labels


class AveBatchSampler(Sampler):
    def __init__(self, data):
        self.data = data.data
        self.batch_size  = data.batch_size
        self.label_distribution=data.label_distribution # 标签分布

    def __iter__(self):
        len_data = len(self.data)
        indices = [i for i in range(len_data)]
        random.shuffle(indices) # 打乱下标
        for _ in range(10000):  # 试错10000
            idx1 = random.randint(0,len_data-1)
            idx2 = random.randint(0,len_data-1)
            if indices[idx1] // self.batch_size == indices[idx2] // self.batch_size:
                continue
            # 判断两个是否有必要交换
            batch_labels1 = [self.data[i][-1].copy().reshape(1,-1) for i in range(indices[idx1]//self.batch_size*self.batch_size,\
                                                    min((indices[idx1]//self.batch_size+1)*self.batch_size,len(self.data)))]
            batch_labels2 = [self.data[i][-1].copy().reshape(1,-1) for i in range(indices[idx2]//self.batch_size*self.batch_size,\
                                                    min((indices[idx2]//self.batch_size+1)*self.batch_size,len(self.data)))]

            label1 = self.data[indices[idx1]][-1].copy()
            label2 = self.data[indices[idx2]][-1].copy()
            batch_labels1 = np.sum(np.concatenate(batch_labels1,axis=0)) # [class_num]
            batch_labels2 = np.sum(np.concatenate(batch_labels2,axis=0)) # [class_num]

            batch_labels1_changed = batch_labels1 + label2 - label1
            batch_labels2_changed = batch_labels2 + label1 - label2

            score1 = np.sum(np.power(batch_labels1/(np.sum(batch_labels1)+1e-3)-self.label_distribution,2)) + \
                     np.sum(np.power(batch_labels2/(np.sum(batch_labels2)+1e-3)-self.label_distribution,2))
            score2 = np.sum(np.power(batch_labels1_changed/(np.sum(batch_labels1_changed)+1e-3)-self.label_distribution,2)) + \
                     np.sum(np.power(batch_labels2_changed/(np.sum(batch_labels2_changed)+1e-3)-self.label_distribution,2))
            if score2 < score1:
                indices[idx1],indices[idx2] = indices[idx2],indices[idx1]
        
        return iter(indices)

    def __len__(self):
        return len(self.data)



def data_loader(data_file, opt, shuffle, num_workers=0):
    dataset = Dataset(data_file, opt)
    if not opt.use_average_batch: # 不使用 平均机制
        loader = data.DataLoader(dataset=dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=shuffle,
                                 pin_memory=True,
                                 num_workers=num_workers,
                                 collate_fn=collate_fn)
    else:
        if shuffle:
            sampler = AveBatchSampler(dataset)
            loader = data.DataLoader(dataset=dataset,
                                     batch_size=opt.batch_size,
                                     sampler=sampler,
                                     pin_memory=True,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn)
        else:
            loader = data.DataLoader(dataset=dataset,
                                     batch_size=opt.batch_size,
                                     shuffle=shuffle,
                                     pin_memory=True,
                                     num_workers=num_workers,
                                     collate_fn=collate_fn)

    return loader
