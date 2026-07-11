"""
MAFSCN完整模型
Multi-Attention Fusion Sentiment Classification Network
基于BERT的多注意力融合情感分类网络
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from src.models.attention import MultiAttentionFusion


class MAFSCN(nn.Module):
    """
    多注意力融合情感分类网络

    架构：
    1. BERT编码器（提取语义特征）
    2. 多注意力融合模块（通道+空间+交叉注意力）
    3. 分类器（全连接层）
    """

    def __init__(self, bert_model_name='./bert-base-chinese',
                 num_classes=3, dropout=0.3, use_multi_layer=False):
        """
        初始化MAFSCN模型

        Args:
            bert_model_name: BERT模型路径或名称
            num_classes: 分类类别数（默认3：负面/中性/正面）
            dropout: Dropout比率
            use_multi_layer: 是否使用BERT多层特征
        """
        super(MAFSCN, self).__init__()

        self.num_classes = num_classes
        self.hidden_size = 768
        self.use_multi_layer = use_multi_layer

        # 1. BERT编码器
        self.bert = BertModel.from_pretrained(bert_model_name)

        # 2. 多注意力融合模块
        self.multi_attention = MultiAttentionFusion(hidden_size=self.hidden_size)

        # 3. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        # 4. 可选：多层特征融合层
        if use_multi_layer:
            self.layer_fusion = nn.Sequential(
                nn.Linear(self.hidden_size * 3, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(self, input_ids, attention_mask):
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len] - 输入token ids
            attention_mask: [batch_size, seq_len] - 注意力掩码

        Returns:
            logits: [batch_size, num_classes] - 分类logits
            weights_dict: dict - 融合权重
            cross_weights: 交叉注意力权重
        """
        # 1. BERT编码
        if self.use_multi_layer:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states

            # 提取浅、中、深三层特征
            F1 = hidden_states[-4]   # 浅层（第9层）词汇级语义
            F2 = hidden_states[-8]   # 中层（第5层）句子级上下文
            F3 = hidden_states[-12]  # 深层（第1层）全局语义

            cls_f1 = F1[:, 0, :]
            cls_f2 = F2[:, 0, :]
            cls_f3 = F3[:, 0, :]

            multi_cls = torch.cat([cls_f1, cls_f2, cls_f3], dim=-1)
            cls_embedding = self.layer_fusion(multi_cls)
            sequence_output = F3
        else:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            sequence_output = outputs.last_hidden_state

        # 2. 多注意力融合
        fused_features, weights_dict, cross_weights = self.multi_attention(
            cls_embedding, sequence_output
        )

        # 3. 分类
        logits = self.classifier(fused_features)

        return logits, weights_dict, cross_weights

    def freeze_bert(self):
        """冻结BERT参数，只训练注意力模块和分类器"""
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert(self):
        """解冻BERT参数"""
        for param in self.bert.parameters():
            param.requires_grad = True