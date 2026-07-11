"""
BERT 多分类模型定义
供推荐服务和训练共用
"""

import torch
import torch.nn as nn
from transformers import BertModel


class BERTMultiClassifier(nn.Module):
    """BERT 多分类模型"""

    def __init__(self, num_classes, bert_model_name='bert-base-chinese', dropout=0.3):
        super(BERTMultiClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits