"""
PyTorch Dataset类
用于加载预处理后的评论数据，供训练和推理使用
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config import DataConfig, PathConfig


class SentimentDataset(Dataset):
    """
    情感分析数据集
    加载预处理CSV，返回 input_ids / attention_mask / labels
    """

    def __init__(self, texts, labels, tokenizer, max_len=None):
        """
        Args:
            texts: 文本列表（分词后的文本）
            labels: 标签列表 (0=负面, 1=中性, 2=正面)
            tokenizer: BERT tokenizer
            max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len or DataConfig.MAX_SEQ_LEN

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class RecommendDataset(Dataset):
    """
    景区推荐数据集
    加载景区标注数据，返回 input_ids / attention_mask / label(景区ID)
    """

    def __init__(self, texts, labels, tokenizer, max_len=None):
        """
        Args:
            texts: 文本列表（分词后的文本）
            labels: 景区ID列表（LabelEncoder编码后的整数）
            tokenizer: BERT tokenizer
            max_len: 最大序列长度
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len or DataConfig.MAX_SEQ_LEN

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def load_sentiment_data(csv_path=None):
    """
    加载预处理后的情感分析数据

    Returns:
        texts: 文本列表
        labels: 标签列表
        df: 原始DataFrame
    """
    if csv_path is None:
        csv_path = PathConfig.DATA_PREPROCESSED

    df = pd.read_csv(csv_path)

    # 使用分词后的文本
    text_col = 'segmented' if 'segmented' in df.columns else '评论内容'
    texts = df[text_col].astype(str).tolist()
    labels = df['情感标签'].tolist() if '情感标签' in df.columns else df['label'].tolist()

    labels_np = np.array(labels)
    print(f"[dataset] 加载情感数据: {len(texts)}条")
    print(f"  负面(0): {(labels_np == 0).sum()}条")
    print(f"  中性(1): {(labels_np == 1).sum()}条")
    print(f"  正面(2): {(labels_np == 2).sum()}条")

    return texts, labels, df


def load_recommend_data(csv_path=None):
    """
    加载景区推荐数据

    Returns:
        texts: 文本列表
        scenic_names: 景区名称列表
        labels: 编码后的标签
        label_encoder: LabelEncoder
        df: 原始DataFrame
    """
    from sklearn.preprocessing import LabelEncoder

    if csv_path is None:
        csv_path = PathConfig.DATA_PREPROCESSED

    df = pd.read_csv(csv_path)

    # 确定景区列名
    scenic_col = '景区' if '景区' in df.columns else '来源景区'
    text_col = 'segmented' if 'segmented' in df.columns else '评论内容'

    texts = df[text_col].astype(str).tolist()
    scenic_names = df[scenic_col].unique().tolist()

    # 标签编码
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df[scenic_col])

    print(f"[dataset] 加载推荐数据: {len(texts)}条, {len(scenic_names)}个景区")

    return texts, scenic_names, labels, label_encoder, df


if __name__ == '__main__':
    texts, labels, df = load_sentiment_data()
    print(f"\n示例文本: {texts[0][:100]}...")
    print(f"示例标签: {labels[0]}")