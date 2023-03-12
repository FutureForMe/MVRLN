# -*- coding:utf-8 -*- 
import json
import random
import os
import numpy as np
import copy
from utils import GenerateEmbedding

data_path = r"Data/split_data"
train_path=os.path.join(data_path,'train.json')
label_path=os.path.join(data_path,'label2id.txt')
label_key = '出院诊断'
bert_path='pretrain_language_models/bert_chinese'
cuda=0

"""
    预处理过程一：为损失函数提供类别出现频率矩阵
"""

with open(label_path,'r',encoding='utf-8') as f:
    labels = f.read().split('\n')
    labels = [label.split(' ')[0] for label in labels]

with open(train_path, "r", encoding="UTF-8") as f:
    train = json.load(f)

label_freq = [1 for _ in range(len(labels))]
for item in train:
    for disease in item[label_key]:
        if disease in labels:
            label_freq[labels.index(disease)] += 1


with open(os.path.join(data_path, "label_freq.txt"),'w',encoding='utf-8') as f:
    f.write(str(label_freq))

    
"""
    预处理过程二：构造标签标签共现矩阵
"""

with open(label_path,'r',encoding='utf-8') as f:
    labels = f.read().split('\n')
    labels = [label.split(' ')[0] for label in labels]

co_array = np.zeros((len(labels),len(labels)))
labels_num = np.zeros((len(labels),1))

with open(train_path,'r',encoding='utf-8')as f:
    train = json.load(f)
    for item in train:
        for label1 in item[label_key]:
            if label1 not in labels:
                continue
            for label2 in item[label_key]:
                if label2 not in labels:
                    continue
                if label1 == label2:
                    labels_num[labels.index(label1)][0] += 1
                else:
                    co_array[labels.index(label1),labels.index(label2)] += 1

co_array /= (labels_num + 3) # 加入平滑
co_array += np.eye(len(labels)) # 加入

np.save(os.path.join(data_path,'label_label_matrix.npy'),co_array)

"""
    预处理过程三：构造药物-实体相关矩阵
"""

entity2id = {}
co_entity_arr = []
entity_arr = []

zeros = [0 for _ in range(len(labels))]

with open(label_path,'r',encoding='utf-8') as f:
    labels = f.read().split('\n')
    labels = [label.split(' ')[0] for label in labels]

with open(train_path,'r',encoding='utf-8')as f:
    train = json.load(f)
    for item in train:
        entities = set(item['疾病实体']) | set(item['治疗实体']) | set(item['检查实体']) | set(item['症状实体']) | set(item['检查结果实体'])
        for entity in entities:
            if entity not in entity2id:
                entity2id[entity] = len(entity2id)
                co_entity_arr.append(copy.copy(zeros))
                entity_arr.append(0)
            entity_arr[entity2id[entity]] += 1
            for label in item[label_key]:
                if label not in labels:
                    continue
                co_entity_arr[entity2id[entity]][labels.index(label)] += 1

co_entity_arr = np.array(co_entity_arr)
entity_arr = np.array(entity_arr).reshape(-1,1) # [entity_num,1]

co_entity_arr = co_entity_arr / (entity_arr + 10) # +10是拉普拉斯平滑
co_entity_arr[co_entity_arr<0.3] = 0              # 采用软平滑
# co_entity_arr[co_entity_arr>=0.3] = 1              # 采用软平滑
# [entity_num, class_num]

np.save(os.path.join(data_path,'ent_label_matrix.npy'),co_entity_arr) # [ent,med]
with open(os.path.join(data_path,'entity2id.txt'),'w',encoding='utf-8') as f:
    f.write('\n'.join([entity for entity in entity2id]))

"""
    预处理过程四：预训练实体向量
"""


gm=GenerateEmbedding(bert_path,cuda)
entities=np.zeros((len(entity2id),768))
for i,entity in enumerate(entity2id):
    entities[i]=gm.generate(entity)
np.save(os.path.join(data_path,'entities.npy'),entities)


