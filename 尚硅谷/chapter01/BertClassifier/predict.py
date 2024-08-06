# coding: utf-8

import torch
from transformers import BertTokenizer, BertConfig
from common import constants
from model import BertClassifier

def process_data(text):
    tokenizer = BertTokenizer.from_pretrained(constants.BERT_PATH)
    token = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)

    input_ids = token['input_ids']
    input_ids = torch.LongTensor(input_ids).unsqueeze(0)
    token_type_ids = token['token_type_ids']
    token_type_ids = torch.LongTensor(token_type_ids).unsqueeze(0)
    attention_mask = token['attention_mask']
    attention_mask = torch.LongTensor(attention_mask).unsqueeze(0)
    
    return input_ids, token_type_ids, attention_mask

# 加载模型

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

bert_config = BertConfig.from_pretrained(constants.BERT_PATH)
num_labels = 10
model = BertClassifier(bert_config, num_labels).to(device)
state_dict = torch.load(constants.MODEL_PATH, map_location=torch.device(device))

# 多卡训练保存的模型在cpu上加载时需要额外处理下
from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] if k.startswith('module.') else k  # 去掉 'module.' 前缀
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)

# 预测
while True:
    
    text = input("input: ")
    
    if text == "byebye":
        break
    
    input_ids, token_type_ids, attention_mask = process_data(text)

    output = model(input_ids.to(device), attention_mask.to(device), token_type_ids.to(device))
    pred_label = torch.argmax(output, dim=1)

    labels = ['体育', '娱乐', '家居', '房产', '教育', '时尚', '时政', '游戏', '科技', '财经']

    print("label:", labels[pred_label[0]])