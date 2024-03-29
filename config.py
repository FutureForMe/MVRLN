import os
import json
import numpy as np

class DefaultConfig(object):

    def __init__(self, args, save_model_path):
        self.train_path = os.path.join(args.data_path, "train.json")
        self.dev_path = os.path.join(args.data_path, "dev.json")
        self.test_path = os.path.join(args.data_path, "test.json")
        self.label_idx_path = os.path.join(args.data_path, "label2id.txt")
        self.label_freq_path = os.path.join(args.data_path,'label_freq.txt')
        self.label_matrix_path = os.path.join(args.data_path,'label_label_matrix.npy')
        self.label_matrix = np.load(self.label_matrix_path)
        self.ent_label_matrix_path = os.path.join(args.data_path,'ent_label_matrix.npy') #[ent_num,label_num]
        self.ent_label_matrix = np.load(self.ent_label_matrix_path)
        self.entity_idx_path = os.path.join(args.data_path,'entity2id.txt')
        self.entities_pretrain_embed=np.load(os.path.join(args.data_path,'entities.npy'))
        for attr in dir(args):
            if attr[:1] == '_':
                continue
            setattr(self,attr,getattr(args,attr))
            
        self.label2id = {}
        with open(self.label_idx_path, "r", encoding="UTF-8")as f:
            for line in f:
                lin = line.strip().split()
                self.label2id[lin[0]] = len(self.label2id)

        self.entity2id = {}
        with open(self.entity_idx_path, "r", encoding="UTF-8")as f:
            for line in f:
                lin = line.strip().split()
                self.entity2id[lin[0]] = len(self.entity2id)

        self.save_model_path = save_model_path
        self.class_num = len(self.label2id)
        self.entity_num= len(self.entity2id)


    def __str__(self):
        ans = "====================Configuration====================\n"
        for key, value in self.__dict__.items():
            if key in ['label2id','entity2id']:
                continue
            ans += key + ":" + (value if type(value) == str else str(value)) + "\n"
        ans += "====================Configuration====================\n"

        return ans
