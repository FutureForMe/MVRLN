import numpy as np
import torch
import re
import copy
from sko.GA import GA

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.sum(mx, dim=1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx


def normalize_adj(adj):
    """D^(-1/2)AD^(-1/2)"""
    D = torch.diag(torch.sum(adj != 0., dim=1)).float()
    D_2 = torch.pow(D, -0.5)
    D_2[torch.isinf(D_2)] = 0.
    ans = torch.mm(D_2, adj)
    ans = torch.mm(ans, D_2)
    return ans


def normalize_features(mx):
    rowsum = mx.sum(1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx


def union_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_or(yhat, y).sum(axis=axis).astype(float)


def intersect_size(yhat, y, axis):
    # axis=0 for label-level union (macro). axis=1 for instance-level
    return np.logical_and(yhat, y).sum(axis=axis).astype(float)


def macro_accuracy(yhat, y):
    num = intersect_size(yhat, y, 0) / (union_size(yhat, y, 0) + 1e-10)
    return np.mean(num)


def macro_precision(yhat, y):
    num = intersect_size(yhat, y, 0) / (yhat.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_recall(yhat, y):
    num = intersect_size(yhat, y, 0) / (y.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(yhat, y):
    prec = macro_precision(yhat, y)
    rec = macro_recall(yhat, y)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def all_macro(yhat, y):
    return macro_accuracy(yhat, y), macro_precision(yhat, y), macro_recall(yhat, y), macro_f1(yhat, y)


def micro_accuracy(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (union_size(yhatmic, ymic, 0) + 1e-10)


def micro_precision(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (yhatmic.sum(axis=0) + 1e-10)


def micro_recall(yhatmic, ymic):
    return intersect_size(yhatmic, ymic, 0) / (ymic.sum(axis=0) + 1e-10)


def micro_f1(yhatmic, ymic):
    prec = micro_precision(yhatmic, ymic)
    rec = micro_recall(yhatmic, ymic)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return f1


def all_micro(yhatmic, ymic):
    return micro_accuracy(yhatmic, ymic), micro_precision(yhatmic, ymic), micro_recall(yhatmic, ymic), micro_f1(yhatmic,
                                                                                                                ymic)


def all_metrics(y_hat, y):
    """
    :param y_hat:
    :param y:

    :return:
    """
    names = ['acc', 'prec', 'rec', 'f1']
    macro_metrics = all_macro(y_hat, y)

    y_mic = y.ravel()
    y_hat_mic = y_hat.ravel()
    micro_metrics = all_micro(y_hat_mic, y_mic)

    metrics = {names[i] + "_macro": macro_metrics[i] for i in range(len(macro_metrics))}
    metrics.update({names[i] + '_micro': micro_metrics[i] for i in range(len(micro_metrics))})

    return metrics


def print_metrics(metrics_test):
    print("\n[MACRO] accuracy, precision, recall, f-measure")
    print("%.4f, %.4f, %.4f, %.4f" %
          (metrics_test["acc_macro"], metrics_test["prec_macro"], metrics_test["rec_macro"], metrics_test["f1_macro"]))

    print("[MICRO] accuracy, precision, recall, f-measure")
    print("%.4f, %.4f, %.4f, %.4f" %
          (metrics_test["acc_micro"], metrics_test["prec_micro"], metrics_test["rec_micro"], metrics_test["f1_micro"]))


def write_result(report, result_path):
    with open(result_path, "w", encoding="UTF-8")as f:
        f.write(report)
    
def get_age(raw_age):
    if '岁' in raw_age or '月' in raw_age or '日' in raw_age or '天' in raw_age:
        year = re.search(r'(\d*?)岁',raw_age)
        month = re.search(r'(\d*?)月',raw_age)
        day = re.search(r'(\d*?)日',raw_age)
        day2 = re.search(r'(\d*?)天',raw_age)

        ans = 0
        if year is None or year.group(1)=='': ans += 0
        else: ans += int(year.group(1))*365
        if month is None or month.group(1)=='': ans += 0
        else: ans += int(month.group(1))*30
        if day is None or day.group(1)=='': ans += 0
        else: ans += int(day.group(1))
        if day2 is None or day2.group(1)=='': ans += 0
        else: ans += int(day2.group(1))
        ans = ans // 365
    else:
        if 'Y' in raw_age:
            raw_age = raw_age.replace('Y','')
        try:
            ans = int(raw_age)
        except:
            ans = -1
    if ans < 0:
        return ''
    elif ans >= 0 and ans < 1:
        return '婴儿'
    elif ans >= 1 and ans <= 6:
        return '童年'
    elif ans >=7 and ans <= 18:
        return '少年'
    elif ans >= 19 and ans <= 30:
        return '青年'
    elif ans >= 31 and ans <= 40:
        return '壮年' 
    elif ans >= 41 and ans <= 55:
        return '中年'
    else:
        return '老年'

def format(entity):
    entity = entity.replace('+','\+').replace('*','\*').replace('.','\.')\
                   .replace('(','\(').replace(')','\)').replace('[','\[')\
                   .replace(']','\[')
    return entity

def remove_neg_entities(document, entities):
    entities = list(set(entities))
    for entity in entities:
        index = document.index(entity)
        if '无' in document[min(0,index-20):index] or '否认' in document[min(0,index-20):index]:
            entities.remove(entity)

        # if re.search(r'(无|(否认))(.{0,10}(、|及|，))*?.{0,5}'+format(entity),document) is not None:
        #     entities.remove(entity)
    return entities

import os
import re

class Match(object):
    """docstring for Match"""
    def __init__(self,file_path,entity_embedding_pretrain_path):
        super(Match, self).__init__()

        self.trie_tree = {} # 字典树
        self.entity2id = {} # 实体到id转化字典
        self.disease2entityid = {} # 疾病对应的实体
        self.end_sign = '__end__'

        # 设置实体嵌入
        self.gm = GenerateEmbedding('./bert_chinese') # 设置bert路径
        self.exist_embed = os.path.exists(entity_embedding_pretrain_path)
        self.entity_embed = []
        
        # 构建字典树
        self.build_tree(file_path)
        self.entity_size = len(self.entity2id) # 实体数量

        # log
        print('实体数量：',self.entity_size)
        print(self.disease2entityid)

        # 保存实体文件
        self.entity_embed = np.array(self.entity_embed) # [entity_num,hidden_size]
        if not self.exist_embed:
            np.savez(entity_embedding_pretrain_path,embedding=self.entity_embed)

    def build_tree(self,file_path):
        
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                entities = line.split('：')[1].split('、')
                disease = line.split('：')[0].replace('- ','')
                entities.append(disease)
                self.disease2entityid[disease] = []
                # 遍历所有实体
                for entity in entities:
                    if entity == '':
                        continue

                    if entity not in self.entity2id:
                        self.entity2id[entity] = len(self.entity2id)
                        # 生成预训练实体嵌入
                        if not self.exist_embed:
                            self.entity_embed.append(self.gm.generate(entity))
                    else:
                        continue

                    self.disease2entityid[disease].append(self.entity2id[entity])

                    # 向字典树中添加实体
                    tree_node = self.trie_tree
                    for word in entity:
                        if tree_node.get(word) is None:
                            tree_node[word] = {}
                        tree_node = tree_node[word]
                    tree_node[self.end_sign] = False



    def find_entities(self,doc):
        doc_tokens = [c for c in doc]
        labels = []
        start_idx = -1
        end_idx = -1
        for idx,token in enumerate(doc_tokens):
            if idx <= end_idx: continue # 避免重复加入
            tree_node = self.trie_tree
            token_idx = idx
            first_flag = True
            start_idx = idx
            end_idx = -1

            while token_idx < len(doc_tokens) and \
                 (doc_tokens[token_idx] in tree_node):
                
                if doc_tokens[token_idx] in tree_node: # 字符
                    tree_node = tree_node[doc_tokens[token_idx]]
                    if first_flag:
                        start_idx = token_idx
                        first_flag = False
                    if self.end_sign in tree_node:
                        end_idx = token_idx

                token_idx += 1 # 

            if start_idx<=end_idx:
                labels.append((start_idx,end_idx)) # 一个实体开始位置和结束位置

        entities = [''.join(doc_tokens[start_idx:end_idx+1]) for (start_idx,end_idx) in labels]
        # entities = remove_neg_entities(doc,entities) # 去掉被否定实体
        entity_ids = set([self.entity2id[entity] for entity in entities])
        entities_onehot = [1 if i in entity_ids else 0 for i in range(self.entity_size)]
        return entities,entities_onehot

class Match3(object):
    """docstring for Match"""
    """
        要求graph中disease顺序必须和label2id一致 最复杂的Match类别
    """
    def __init__(self,file_path,entity_embedding_pretrain_path):
        super(Match3, self).__init__()

        self.trie_tree = {} # 字典树
        self.entity2id = {} # 实体到id转化字典
        self.id2entity = [] # id到实体的字典
        self.disease2entityid = {} # 疾病对应的实体
        self.disease2entities = {} # 疾病对应的实体名称
        self.disease2reentities = {} # 疾病到正则
        self.class_num = 52
        self.end_sign = '__end__'
        # 设置实体嵌入
        self.gm = GenerateEmbedding('./bert_chinese') # 设置bert路径
        self.exist_embed = os.path.exists(entity_embedding_pretrain_path)
        self.entity_embed = []

        self.build_tree(file_path)
        self.entity_size = len(self.entity2id) # 实体数量 固定实体和正则实体都加入进来
        self.id2entity = [entity for entity in self.entity2id]
        self.entity_embed = np.array(self.entity_embed) # [entity_num,hidden_size]
        print('实体数量：',self.entity_size)
        print(self.id2entity)
        print(self.disease2entityid)
        if not self.exist_embed:
            np.savez(entity_embedding_pretrain_path,embedding=self.entity_embed)

    def build_tree(self,file_path):
        
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                entities = line.split('：')[1].split('、')
                disease = line.split('：')[0]
                # entities.append(disease)
                self.disease2entityid[disease] = []
                self.disease2entities[disease] = []
                self.disease2reentities[disease] = []
                for entity in entities:
                    if entity == '':
                        continue
                    if entity[0] == '[' and entity[-1] == ']': # 正则
                        self.disease2reentities[disease].append(entity)

                    if entity not in self.entity2id:
                        self.entity2id[entity] = len(self.entity2id)
                        self.entity2id['无'+entity] = len(self.entity2id)
                        if not self.exist_embed:
                            if entity[0] == '[' and entity[-1] == ']':
                                entity_raw = re.sub(r'[\.\*\{\}\,]','',entity[1:-1])
                            else:
                                entity_raw = entity
                            self.entity_embed.append(self.gm.generate(entity_raw))
                            self.entity_embed.append(self.gm.generate('无'+entity_raw)) # 同时添加无的嵌入
                    else:
                        continue

                    self.disease2entityid[disease].append(self.entity2id[entity]) # 把无也加入进来
                    self.disease2entities[disease].append(entity)
                    self.disease2entityid[disease].append(self.entity2id['无'+entity])
                    self.disease2entities[disease].append('无'+entity)

                    if entity[0] == '[' and entity[-1] == ']':
                        continue

                    tree_node = self.trie_tree
                    for word in entity:
                        if tree_node.get(word) is None:
                            tree_node[word] = {}
                        tree_node = tree_node[word]
                    tree_node[self.end_sign] = False
        
    
    def find_entities(self,doc):
        """
            返回值
            entities [entity_num]列表，字符串
            entity2idx {entity:[1,2,3]} # entity到句子下标的矩阵
            entity_array [seq_len,disease_num] 的矩阵
        """
        doc_tokens = [c for c in doc]
        labels = []
        start_idx = -1
        end_idx = -1
        for idx,token in enumerate(doc_tokens):
            if idx <= end_idx: continue # 避免重复加入
            tree_node = self.trie_tree
            token_idx = idx
            first_flag = True
            start_idx = idx
            end_idx = -1

            while token_idx < len(doc_tokens) and \
                 (doc_tokens[token_idx] in tree_node):
                
                if doc_tokens[token_idx] in tree_node: # 字符
                    tree_node = tree_node[doc_tokens[token_idx]]
                    if first_flag:
                        start_idx = token_idx
                        first_flag = False
                    if self.end_sign in tree_node:
                        end_idx = token_idx

                if start_idx <= end_idx:
                    labels.append((start_idx,end_idx)) # 一个实体开始位置和结束位置
                token_idx += 1 # 


        entities = [''.join(doc_tokens[start_idx:end_idx+1]) for (start_idx,end_idx) in labels] # 固定实体列表


        # 正则表达式实体处理
        document = ''.join([c[0] for c in doc_tokens])

        # 获取电子病历中的所有实体
        neg_entities = re.findall(r'((否认))(([^(出现)]{0,10})(、|及))*?(.{0,10})(，|。|；)',document)
        # 否认“肝炎，伤寒，结核”
        neg_entities.extend(re.findall(r'否认“(.*?)”',document))
        neg_entities = ''.join([''.join(list(item)) for item in neg_entities]) # 电子病历中的否定实体

        entities = ['无' + entity if entity in neg_entities else entity for entity in entities] # 无病历

        for disease_id,disease in enumerate(self.disease2reentities):
            for re_entity in self.disease2reentities[disease]:
                match_entities = re.findall(re_entity[1:-1],document)
                for entity in match_entities:
                    if entity not in entities:
                        if entity in neg_entities:
                            entities.append('无' + re_entity)
                        else:
                            entities.append(re_entity)
                        break

        entity_ids = set([self.entity2id[entity] for entity in entities])
        entities_onehot = [1 if i in entity_ids else 0 for i in range(self.entity_size)]
        
        return entities,entities_onehot


### 别名采样
import numpy as np
def create_alias_table(area_ratio):
    """
    area_ratio[i]代表事件i出现的概率
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    N = len(area_ratio)
    accept, alias = [0] * N, [0] * N
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * N
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
            (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias

def alias_sample(accept, alias):
    """
    
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

from transformers import AutoModel,AutoTokenizer
import torch
import numpy as np

class GenerateEmbedding:
    def __init__(self,bert_path,cuda):
        self.cuda = cuda
        self.bert = AutoModel.from_pretrained(bert_path)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_path)
        if torch.cuda.is_available():
            self.bert = self.bert.cuda(cuda)
    def generate(self,entity):
        """
            生成实体嵌入向量
        """
        entity = '#' + entity
        tokens = self.tokenizer(entity,return_tensors = 'pt')
        with torch.no_grad():
            if torch.cuda.is_available():
                tokens['input_ids'] = tokens['input_ids'].cuda(self.cuda)
                tokens['attention_mask'] = tokens['attention_mask'].cuda(self.cuda)
            output = self.bert(tokens['input_ids'],tokens['attention_mask']).last_hidden_state[:,2:]
        return output.squeeze(0).mean(dim = 0).cpu().numpy()

    def similarity(self,vec1,vec2):
        """
            计算余弦相似度
        """
        return np.sum(vec1 * vec2) / (np.sqrt(np.sum(np.power(vec1,2))) + np.sqrt(np.sum(np.power(vec2,2))))



class Match2(object):
    """docstring for Match"""
    def __init__(self,file_path):
        super(Match2, self).__init__()

        self.trie_tree = {} # 字典树
        self.entity2id = {} # 实体到id转化字典
        self.disease2entityid = {} # 疾病对应的实体
        self.end_sign = '__end__'
        self.build_tree(file_path)
        self.entity_size = len(self.entity2id) # 实体数量
        print('实体数量：',self.entity_size)

    def build_tree(self,file_path):
        
        with open(file_path,'r',encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == '':
                    continue
                entities = line.split('：')[1].split('、')
                disease = line.split('：')[0].replace('- ','')
                entities.append(disease)
                self.disease2entityid[disease] = []
                for entity in entities:
                    if entity == '':
                        continue
                    if entity not in self.entity2id:
                        self.entity2id[entity] = len(self.entity2id)
                    else:
                        continue
                    self.disease2entityid[disease].append(self.entity2id[entity])
                    tree_node = self.trie_tree
                    for word in entity:
                        if tree_node.get(word) is None:
                            tree_node[word] = {}
                        tree_node = tree_node[word]
                    tree_node[self.end_sign] = False
    
    def find_entities(self,doc):
        doc_tokens = [c for c in doc]
        labels = []
        start_idx = -1
        end_idx = -1
        for idx,token in enumerate(doc_tokens):
            if idx <= end_idx: continue # 避免重复加入
            tree_node = self.trie_tree
            token_idx = idx
            first_flag = True
            start_idx = idx
            end_idx = -1

            while token_idx < len(doc_tokens) and \
                 (doc_tokens[token_idx] in tree_node):
                
                if doc_tokens[token_idx] in tree_node: # 字符
                    tree_node = tree_node[doc_tokens[token_idx]]
                    if first_flag:
                        start_idx = token_idx
                        first_flag = False
                    if self.end_sign in tree_node:
                        end_idx = token_idx

                token_idx += 1 # 

            if start_idx<=end_idx:
                labels.append((start_idx,end_idx)) # 一个实体开始位置和结束位置

        entities = [''.join(doc_tokens[start_idx:end_idx+1]) for (start_idx,end_idx) in labels]
        # entities = remove_neg_entities(doc,entities) # 去掉被否定实体
        entity_ids = set([self.entity2id[entity] for entity in entities])
        entities_onehot = [1 if i in entity_ids else 0 for i in range(self.entity_size)]
        return entities,entities_onehot


# def best_threshold(Y,Y_hat,prec):
#     """
#         通过验证集确定最佳阈值
#         Y_hat : [batch_size, class_num] ∈ {0,1}
#         Y     : [batch_size, class_num] ∈ [0,1]
#         prec  : float ∈ [0,1] 精度
#         return: [class_num] ∈ [0,1] 最佳阈值
#     """
#     steps = int(1 / prec)
#     best_thresholds = np.zeros((Y.shape[1]))
#     ans = np.zeros((Y.shape[1],steps)) # [class_num, steps]
#     for class_i in range(Y_hat.shape[1]):
#         max_f1 = -1
#         best_threshold = -1
#         for step in range(steps):
#             threshold = step / steps
#             tmp = copy.copy(Y_hat[:,class_i])
#             tmp[tmp > threshold] = 1
#             tmp[tmp <= threshold] = 0
#             p = np.sum(tmp * Y[:,class_i]) / (np.sum(tmp) + 1e-10)
#             r = np.sum(tmp * Y[:,class_i]) / (np.sum(Y[:,class_i]) + 1e-10)
#             f1 = 2 * p * r /(p + r + 1e-10)
#             ans[class_i,step] = f1
#         best_thresholds[class_i] = best_threshold

#     # 1 / steps 进行平滑
#     ans_all = np.zeros((steps//10,Y.shape[1],steps))
#     for i in range(steps//10):
#         offset = i - steps//20
#         ans_all[i,:,max(0,0+offset):min(steps,steps+offset)] = ans[:,max(0,0+offset):min(steps,steps+offset)]

#     ans_smooth = np.sum(ans_all,axis = 0)

#     best_thresholds = np.argmax(ans_smooth,axis = 1).astype(np.float32) # [class_num]
#     best_thresholds /= steps

#     return best_thresholds



def best_threshold(Y,Y_hat,prec):
    """
        通过验证集确定最佳阈值
        Y     : [batch_size, class_num] ∈ {0,1}
        Y_hat : [batch_size, class_num] ∈ [0,1]
        prec  : float ∈ [0,1] 精度
        return: [class_num] ∈ [0,1] 最佳阈值
    """
    def func(threshold):
        threshold = np.expand_dims(threshold, axis=0)
        y_hat = Y_hat.copy()
        y_hat[y_hat>threshold] = 1
        y_hat[y_hat<=threshold] = 0
        return -micro_f1(y_hat.ravel(),Y.ravel()) # + 0.2 * np.mean(np.abs(threshold-0.5)) # 加入正则

    ga = GA(func=func, n_dim=Y.shape[1], size_pop=100, max_iter=500, prob_mut=0.01,
            lb=[0.4]*Y.shape[1], ub=[0.6]*Y.shape[1], precision=[prec]*Y.shape[1])

    best_x, best_y = ga.run()
    return best_x
