# coding: utf-8

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
from common import constants

class CNewsDataset(Dataset):  # 标签 文本
    def __init__(self, filename):
        # 数据集初始化
        self.labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']
        self.labels_id = list(range(len(self.labels)))
        self.tokenizer = BertTokenizer.from_pretrained(constants.BERT_PATH)
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_id = []
        self.load_data(filename)
    
    def load_data(self, filename):
        # 加载数据
        print('loading data from:', filename)
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        for line in tqdm(lines, ncols=100):
            label, text = line.strip().split('\t')
            label_id = self.labels.index(label)
            token = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.label_id.append(label_id)

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)

