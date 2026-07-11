"""
情感分析服务 - 双模型融合推理
MAFSCN v1(中性敏感) + v2(正负精准) 加权融合
"""

import threading
import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer

from config import PathConfig, ModelConfig, DataConfig
from src.models.mafscn import MAFSCN

# 全局缓存
_df = None
_tokenizer = None
_model1 = None
_model2 = None
_loaded = False
_lock = threading.Lock()

LABEL_MAP = {0: '负面', 1: '中性', 2: '正面'}


def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_models():
    """加载双模型+tokenizer+数据，返回(df, tokenizer, model1, model2)"""
    global _df, _tokenizer, _model1, _model2, _loaded
    if _loaded:
        return _df, _tokenizer, _model1, _model2
    with _lock:
        if _loaded:
            return _df, _tokenizer, _model1, _model2
        device = _get_device()
        print("[sentiment_service] 正在加载双模型...")

        # 数据
        _df = pd.read_csv(PathConfig.DATA_PREPROCESSED)

        # Tokenizer
        _tokenizer = BertTokenizer.from_pretrained(PathConfig.BERT_MODEL_NAME)

        # 模型v1
        _model1 = MAFSCN(
            bert_model_name=PathConfig.BERT_MODEL_NAME,
            num_classes=ModelConfig.NUM_CLASSES
        ).to(device)
        if PathConfig.MODEL_V1_PATH and __import__('os').path.exists(PathConfig.MODEL_V1_PATH):
            _model1.load_state_dict(torch.load(PathConfig.MODEL_V1_PATH, map_location=device))
            print(f"[sentiment_service] v1加载成功")
        _model1.eval()

        # 模型v2
        _model2 = MAFSCN(
            bert_model_name=PathConfig.BERT_MODEL_NAME,
            num_classes=ModelConfig.NUM_CLASSES
        ).to(device)
        if PathConfig.MODEL_V2_PATH and __import__('os').path.exists(PathConfig.MODEL_V2_PATH):
            _model2.load_state_dict(torch.load(PathConfig.MODEL_V2_PATH, map_location=device))
            print(f"[sentiment_service] v2加载成功")
        _model2.eval()

        _loaded = True
        print("[sentiment_service] 双模型加载完成")
        return _df, _tokenizer, _model1, _model2


def get_probs(text, tokenizer, model):
    """单模型推理，返回numpy概率数组[P(负),P(中),P(正)]"""
    device = _get_device()
    encoding = tokenizer(
        text, truncation=True, padding='max_length',
        max_length=DataConfig.MAX_SEQ_LEN, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits, _, _ = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)

    return probs[0].cpu().numpy()


def predict_ensemble(text):
    """双模型加权融合预测，返回完整结果字典"""
    df, tokenizer, model1, model2 = load_models()

    probs1 = get_probs(text, tokenizer, model1)
    probs2 = get_probs(text, tokenizer, model2)

    # 按类别加权融合
    w = ModelConfig.ENSEMBLE_WEIGHTS
    fused = np.array([
        w['negative'][0] * probs1[0] + w['negative'][1] * probs2[0],
        w['neutral'][0] * probs1[1] + w['neutral'][1] * probs2[1],
        w['positive'][0] * probs1[2] + w['positive'][1] * probs2[2],
    ])
    fused = fused / np.sum(fused)  # 归一化

    # 决策规则
    threshold = ModelConfig.NEUTRAL_THRESHOLD
    if fused[1] >= threshold:
        pred = 1
    else:
        pred = 0 if fused[0] > fused[2] else 2

    # 各模型独立判断
    pred1 = int(np.argmax(probs1))
    pred2 = int(np.argmax(probs2))

    return {
        'sentiment': LABEL_MAP[pred],
        'confidence': round(float(fused[pred]), 4),
        'probabilities': {
            '负面': round(float(fused[0]), 4),
            '中性': round(float(fused[1]), 4),
            '正面': round(float(fused[2]), 4),
        },
        'model1_result': {
            'sentiment': LABEL_MAP[pred1],
            'confidence': round(float(probs1[pred1]), 4),
            'probabilities': {
                '负面': round(float(probs1[0]), 4),
                '中性': round(float(probs1[1]), 4),
                '正面': round(float(probs1[2]), 4),
            },
        },
        'model2_result': {
            'sentiment': LABEL_MAP[pred2],
            'confidence': round(float(probs2[pred2]), 4),
            'probabilities': {
                '负面': round(float(probs2[0]), 4),
                '中性': round(float(probs2[1]), 4),
                '正面': round(float(probs2[2]), 4),
            },
        },
    }


def analyze_batch(texts):
    """批量分析，返回结果列表"""
    results = []
    for text in texts:
        try:
            if not text or not text.strip():
                results.append({
                    'sentiment': '中性', 'confidence': 0.0,
                    'probabilities': {'负面': 0.0, '中性': 1.0, '正面': 0.0},
                    'model1_result': {'sentiment': '中性', 'confidence': 0.0,
                                      'probabilities': {'负面': 0.0, '中性': 1.0, '正面': 0.0}},
                    'model2_result': {'sentiment': '中性', 'confidence': 0.0,
                                      'probabilities': {'负面': 0.0, '中性': 1.0, '正面': 0.0}},
                })
            else:
                results.append(predict_ensemble(text))
        except Exception:
            results.append({
                'sentiment': '错误', 'confidence': 0.0,
                'probabilities': {'负面': 0.0, '中性': 0.0, '正面': 0.0},
                'model1_result': {'sentiment': '错误', 'confidence': 0.0,
                                  'probabilities': {'负面': 0.0, '中性': 0.0, '正面': 0.0}},
                'model2_result': {'sentiment': '错误', 'confidence': 0.0,
                                  'probabilities': {'负面': 0.0, '中性': 0.0, '正面': 0.0}},
            })
    return results