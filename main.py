# -*- coding: utf-8 -*-
from importlib import import_module

import torch
import torch.nn as nn
import os
import argparse

import transformers
import warnings
import numpy as np
from sklearn import metrics
import random
import time
from tqdm import tqdm
import json
import datetime

from config import DefaultConfig

from Module.Gradient_Attack import FGM, PGD
from utils import all_metrics, print_metrics, write_result
from util_loss import ResampleLoss,L2loss_func
import re
from utils import best_threshold

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
warnings.filterwarnings('ignore')

def train(opt, train_data_loader, dev_data_loader,test_data_loader,k_fold=0):
    global adv_model, K
    random.seed(opt.seed)
    os.environ['PYTHONHASHSEED'] = str(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True

    model = import_module('Models.' + opt.model_name).Model(opt)
    if torch.cuda.is_available():
        model = model.cuda(opt.gpu)
    train_num = len(train_data_loader) * opt.batch_size
    with open(opt.label_freq_path,'r',encoding='utf-8') as f:
        class_freq = eval(f.read())

    if opt.gradient_att == "FGM":
        adv_model = FGM(model, epsilon=0.5, emb_name='word_embedding')

    if opt.loss_func == 'BCE':
        loss_func = nn.BCEWithLogitsLoss()
        
    if opt.loss_func == 'FL':
        loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 class_freq=class_freq, train_num=train_num) 
        
    if opt.loss_func == 'CBloss': #CB
        loss_func = ResampleLoss(reweight_func='CB', loss_weight=5.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                 class_freq=class_freq, train_num=train_num) 
        
    if opt.loss_func == 'R-BCE-Focal': # R-FL
        loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                                 class_freq=class_freq, train_num=train_num)
        
    if opt.loss_func == 'NTR-Focal': # NTR-FL
        loss_func = ResampleLoss(reweight_func=None, loss_weight=0.5,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 class_freq=class_freq, train_num=train_num)  

    if opt.loss_func == 'DBloss-noFocal': # DB-0FL
        loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=0.5,
                                 focal=dict(focal=False, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                                 class_freq=class_freq, train_num=train_num)
        
    if opt.loss_func == 'CBloss-ntr': # CB-NTR
        loss_func = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                 class_freq=class_freq, train_num=train_num)
        
    if opt.loss_func == 'DBloss': # DB
        loss_func = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                 map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                                 class_freq=class_freq, train_num=train_num)

    bert_params = set(model.word_embedding.parameters())
    other_params = list(set(model.parameters()) - bert_params)
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [
        {'params': [p for n, p in model.word_embedding.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': opt.bert_lr,
         'weight_decay': 1e-2},
        {'params': [p for n, p in model.word_embedding.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': 0.0,
         'weight_decay': 0.0},
        {'params': other_params,
         'lr': opt.other_lr,
         'weight_decay': 0}
    ]

    updates_total = len(train_data_loader) // (opt.accumulation_steps) * opt.epochs
    optimizer = transformers.AdamW(param_optimizer, lr=opt.other_lr, weight_decay=0.0)
    # optimizer = torch.optim.Adam(model.parameters(),lr = opt.other_lr)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=opt.warmup_rate * updates_total,
                                                             num_training_steps=updates_total)

    max_micro_f1 = -1.0  # the best micro F1
    max_scores = [-1 for _ in range(opt.class_num)]
    no_imp_valid = 0  # patience no improvement
    for epoch in range(opt.epochs):
        print("\n=== Epoch %d train ===" % epoch)
        for i, data in enumerate(tqdm(train_data_loader)):
            model.train()
            if torch.cuda.is_available():
                for d in range(len(data)):
                    data[d] = data[d].cuda(opt.gpu)
            input_idxes0,input_idxes1, input_idxes2,input_idxes3,\
            mask0,mask1, mask2,mask3, entities, label_matrix,ent_label_matrix, labels = data
            if 'MultiViewModel' in opt.model_name and 'Longformer' not in opt.bert_path:
                output = model((input_idxes0, input_idxes1, input_idxes2),
                               (mask0, mask1, mask2),entities,label_matrix,ent_label_matrix)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            elif 'MultiViewModel' in opt.model_name and 'Longformer' in opt.bert_path:
                output = model((input_idxes3,),(mask3,),entities,label_matrix,ent_label_matrix)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            elif opt.model_name=='BertModel':
                output = model((input_idxes0, input_idxes1, input_idxes2),
                               (mask0, mask1, mask2))
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            elif opt.model_name=='LongFormerModel' or opt.model_name=='TextRNN':
                output = model(input_idxes3,mask3)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            elif opt.model_name=='TextCNN' or opt.model_name=='TextRCNN' or \
                 opt.model_name=='CAML' or opt.model_name=='MultiResCNN' or \
                 opt.model_name=='LAAT':
                output = model(input_idxes3)
                loss = (loss_func(output, labels)) / opt.accumulation_steps
            else:
                raise Exception
            loss.backward()

            if opt.gradient_att == "FGM":
                adv_model.attack()
                if 'MultiViewModel' in opt.model_name and 'Longformer' not in opt.bert_path:
                    output_adv = model((input_idxes0, input_idxes1, input_idxes2),
                                   (mask0, mask1, mask2),entities,label_matrix,ent_label_matrix)
                elif 'MultiViewModel' in opt.model_name and 'Longformer' in opt.bert_path:
                    output_adv = model((input_idxes3,),(mask3,),entities,label_matrix,ent_label_matrix)
                elif opt.model_name=='BertModel':
                    output_adv = model((input_idxes0, input_idxes1, input_idxes2),
                                   (mask0, mask1, mask2))
                elif opt.model_name=='LongFormerModel' or opt.model_name=='TextRNN':
                    output_adv = model(input_idxes3,mask3)
                    loss = (loss_func(output, labels)) / opt.accumulation_steps
                elif opt.model_name=='TextCNN' or opt.model_name=='TextRCNN' or \
                     opt.model_name=='CAML' or opt.model_name=='MultiResCNN' or \
                     opt.model_name=='LAAT':
                    output_adv = model(input_idxes3)
                loss_adv = loss_func(output_adv, labels)
                loss_adv.backward()
                adv_model.restore()
            if (i+1) % opt.accumulation_steps == 0: 
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # if (i+1) % 600 == 0:
            #     valid_macro_f1, report,scores = inference(model, dev_data_loader,opt,k_fold,batch_no = i//opt.accumulation_steps + updates_total//opt.epochs*epoch)
            #     if valid_macro_f1 > max_macro_f1:
            #         max_macro_f1 = valid_macro_f1
            #         torch.save(model.state_dict(),opt.save_model_path)
            #         print("目前最优验证集结果:{:.5f}".format(max_macro_f1))

        print(f'epochs {epoch} end')
        valid_micro_f1, report,best_threshold = inference(model, dev_data_loader,opt,k_fold)
        test_micro_f1, report,_ = inference(model, test_data_loader,opt,k_fold,thresholds = best_threshold)


        print("\n验证集micro f1: {:.5f}".format(valid_micro_f1))
        print("\n测试集micro f1: {:.5f}".format(test_micro_f1))
        if valid_micro_f1 > max_micro_f1:
            max_micro_f1 = valid_micro_f1

            torch.save({'dict':model.state_dict(),'thresholds':best_threshold},opt.save_model_path)

        print("目前最优验证集结果:{:.5f}".format(max_micro_f1))
        print("\n=== Epoch %d end ===" % epoch)


def inference(model, data_loader, opt, k_fold=0,test_set = False,thresholds=None):
    """validation"""
    model.eval()
    y, y_hat = [], []
    with torch.no_grad():
        for ii, data in enumerate(data_loader):
            if torch.cuda.is_available():
                for d in range(len(data)):
                    data[d] = data[d].cuda(opt.gpu)
            input_idxes0,input_idxes1, input_idxes2,input_idxes3, \
            mask0,mask1, mask2,mask3,entities, label_matrix,ent_label_matrix, labels = data
            if 'MultiViewModel' in opt.model_name and 'Longformer' not in opt.bert_path:
                output = model((input_idxes0, input_idxes1, input_idxes2),
                               (mask0, mask1, mask2),entities,label_matrix,ent_label_matrix)
            elif 'MultiViewModel' in opt.model_name and 'Longformer' in opt.bert_path:
                output = model((input_idxes3,),(mask3,),entities,label_matrix,ent_label_matrix)
            elif opt.model_name=='BertModel':
                output = model((input_idxes0, input_idxes1, input_idxes2),
                                           (mask0, mask1, mask2))
            elif opt.model_name=='LongFormerModel' or opt.model_name=='TextRNN':
                output = model(input_idxes3,mask3)
            elif opt.model_name=='TextCNN' or opt.model_name=='TextRCNN' or \
                 opt.model_name=='CAML' or opt.model_name=='MultiResCNN' or \
                 opt.model_name=='LAAT':
                output = model(input_idxes3)
            else:
                raise Exception
            output = torch.sigmoid(output)

            labels = labels.data.cpu().numpy()
            output = output.data.cpu().numpy()

            y.append(labels)
            y_hat.append(output)

    y = np.concatenate(y, axis=0)
    y_hat = np.concatenate(y_hat, axis=0)
    y = np.round(y)

    y_hat2 = y_hat.copy()
    if opt.use_dynamic_threshold:
        if thresholds is None:
            thresholds = best_threshold(y,y_hat,0.01)
        y_hat[y_hat2>thresholds] = 1
        y_hat[y_hat2<=thresholds] = 0
    else:
        y_hat[y_hat2>0.5] = 1
        y_hat[y_hat2<=0.5] = 0
    id2labels = []
    with open(opt.label_idx_path, "r", encoding="utf-8") as f:
        for line in f:
            lin = line.strip().split()
            id2labels.append(lin[0])
    if test_set:
        with open(opt.test_path,'r',encoding='utf-8') as f:
            data = json.load(f)
            emr_ids = [item['病历编号'] for item in data]

        err_data = []
        for i in range(y.shape[0]):
            if np.logical_or(y_hat[i], y[i]).sum(axis=0) == np.logical_and(y_hat[i], y[i]).sum(axis=0):
                continue
            emr_id = emr_ids[i]
            y_hat_labels = []
            y_labels = []
            for j in range(y.shape[1]):
                if y_hat[i,j] == 1:
                    y_hat_labels.append(id2labels[j])
                if y[i,j] == 1:
                    y_labels.append(id2labels[j])
            item = data[i]
            item['错误输出'] = y_hat_labels
            err_data.append(item)
            # line = ','.join([emr_id,'|'.join(y_labels),'|'.join(y_hat_labels)])
            # table.append(line)

    metrics_test = all_metrics(y_hat, y)
    print_metrics(metrics_test)
    report = metrics.classification_report(y, y_hat, digits=4,target_names = id2labels)
    print(report)

    if not test_set:
        return metrics_test["f1_micro"], report, thresholds
    else:
        return metrics_test["f1_micro"], report, json.dumps(err_data,ensure_ascii=False,indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--model_name', type=str, default='BertCNN_v1')
    parser.add_argument('--bert_path', type=str, default='bert_chinese', help='pretrained path')
    parser.add_argument('--data_path', type=str, default="Data/split_data")
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--dilated', action="store_true", default=False, help="Dilated CNN")
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--gradient_att', type=str, default=None, choices=['FGM', 'PGD', None])
    parser.add_argument('--accumulation_steps',type=int ,default = 1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--bert_lr', type=float, default=1e-5)
    parser.add_argument('--other_lr', type=float, default=5e-4)
    parser.add_argument('--warmup_rate', type=float, default=0.3)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--result_path', type=str, default="result")
    parser.add_argument('--entity_num', type=int, default=34)
    parser.add_argument('--fuse', type=str, default='CAT')
    parser.add_argument('--gpu', type=int, default=1)

    # 隐藏数
    parser.add_argument('--hidden_size',type=int,default=768)
    # 损失函数
    parser.add_argument('--loss_func', type=str, default='BCE')
    parser.add_argument('--fl_alpha',type=float,default=0.7)
    # 多任务
    parser.add_argument('--mid_lambda', type=float, default=0.5)
    # 使用 treatment
    parser.add_argument('--use_treatment',default=False,action='store_true')
    # 使用KNN
    parser.add_argument('--knn_k',type = int,default = 3)
    parser.add_argument('--knn_lambda',type = float,default=1)
    # 标签平滑
    parser.add_argument('--label_smooth_lambda',type=float,default=0)
    # rnn
    parser.add_argument('--rnn',type=str,default='lstm',choices=['lstm','gru'])
    # k折交叉
    parser.add_argument('--k_fold',type=int,default=1)
    # 使用动态阈值
    parser.add_argument('--use_dynamic_threshold',default=False,action='store_true')
    # 平均batch
    parser.add_argument('--use_average_batch',default=False,action='store_true')


    args = parser.parse_args()

    save_model_names = [args.bert_path.split('/')[-1], args.model_name, str(args.gradient_att), "seed", str(args.seed),
                        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")]
    save_model_path = os.path.join("checkpoints", '_'.join(save_model_names) + ".pth")     # best model path
    result_path = os.path.join("result", '_'.join(save_model_names) + ".txt")         # the report of test dataset path
    error_path = os.path.join("result", '_'.join(save_model_names) + "err.txt")         # the report of test dataset path
    score_path = os.path.join("result",'_'.join(save_model_names) + "score.txt")      # 保存分数获取结果
    correct_path = os.path.join(args.data_path,'train_correct.json')                  # the correct of test dataset path

    opt = DefaultConfig(args, save_model_path)
    
    from Dataloader.Data_loader import data_loader

    print(opt)

    train_data_loader = data_loader(opt.train_path, opt, shuffle=True)
    dev_data_loader = data_loader(opt.dev_path, opt, shuffle=False)
    test_data_loader = data_loader(opt.test_path, opt, shuffle=False)
    train(opt, train_data_loader, dev_data_loader,test_data_loader)
    model = import_module('Models.' + opt.model_name).Model(opt)
    if torch.cuda.is_available():
        model = model.cuda(opt.gpu)
    save_dict = torch.load(opt.save_model_path)
    model.load_state_dict(save_dict['dict'])
    thresholds = save_dict['thresholds']
    micro_f1, report,_ = inference(model, dev_data_loader, opt,thresholds = thresholds)
    micro_f1, report,error_report = inference(model, test_data_loader, opt,thresholds = thresholds,test_set=True)
    print('测试集micro-f1',micro_f1)
    write_result(report, result_path)
    write_result(error_report, error_path)

    print("==============Finish==============")
