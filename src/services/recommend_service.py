"""
景区推荐服务 - BERT多分类推荐
"""

import os
import threading
import torch
import pandas as pd
import joblib
from transformers import BertTokenizer

from config import PathConfig, DataConfig
from src.models.bert_classifier import BERTMultiClassifier

# 全局缓存
_tokenizer = None
_model = None
_scenic_names = None  # 按LabelEncoder排序的景区名称列表
_scenic_info = None
_scenic_keywords = None
_loaded = False
_lock = threading.Lock()


def _get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_scenic_info():
    """加载景区信息xlsx"""
    info_path = PathConfig.SCENIC_INFO
    if not os.path.exists(info_path):
        return {}
    df = pd.read_excel(info_path)
    info = {}
    for _, row in df.iterrows():
        name = str(row.iloc[0]).strip()
        folder = DataConfig.SCENIC_IMAGE_MAP.get(name, name)
        info[name] = {
            'address': str(row.iloc[1]) if pd.notna(row.iloc[1]) else '',
            'price': str(row.iloc[2]) if pd.notna(row.iloc[2]) else '',
            'reason': str(row.iloc[3]) if pd.notna(row.iloc[3]) else '',
            'intro': str(row.iloc[4]) if len(row) > 4 and pd.notna(row.iloc[4]) else '',
            'hours': str(row.iloc[5]) if len(row) > 5 and pd.notna(row.iloc[5]) else '',
            'policy': str(row.iloc[6]) if len(row) > 6 and pd.notna(row.iloc[6]) else '',
            'phone': str(row.iloc[7]) if len(row) > 7 and pd.notna(row.iloc[7]) else '',
            'image': f'/data/scenic_image/{folder}/1.jpg',
            'image2': f'/data/scenic_image/{folder}/2.jpg',
        }
    return info


def _load_scenic_keywords():
    """加载景区关键词csv（如存在）"""
    kw_path = os.path.join(PathConfig.DATA_PROCESSED_DIR, 'scenic_keywords.csv')
    if not os.path.exists(kw_path):
        return {}
    kw_df = pd.read_csv(kw_path)
    keywords = {}
    for _, row in kw_df.iterrows():
        name = str(row.iloc[0]).strip()
        keywords[name] = {
            'pos_keywords': str(row.iloc[1]) if pd.notna(row.iloc[1]) else '',
            'neg_keywords': str(row.iloc[2]) if pd.notna(row.iloc[2]) else '',
        }
    return keywords


def load_model():
    """加载推荐模型+tokenizer+景区数据"""
    global _tokenizer, _model, _scenic_names, _scenic_info, _scenic_keywords, _loaded
    if _loaded:
        return _tokenizer, _model, _scenic_names, _scenic_info, _scenic_keywords
    with _lock:
        if _loaded:
            return _tokenizer, _model, _scenic_names, _scenic_info, _scenic_keywords
        device = _get_device()
        print("[recommend_service] 正在加载推荐模型...")

        # 景区名称列表 —— 必须与训练时LabelEncoder的排序一致（按拼音排序）
        # 训练代码: label_encoder = LabelEncoder(); labels = label_encoder.fit_transform(df['来源景区'])
        # LabelEncoder按字母/拼音排序生成classes_
        preproc_df = pd.read_csv(PathConfig.DATA_PREPROCESSED)
        scenic_col = '景区' if '景区' in preproc_df.columns else '来源景区'
        all_scenic = preproc_df[scenic_col].dropna().unique().tolist()
        _scenic_names = sorted(all_scenic)  # 按拼音排序，与LabelEncoder一致

        # Tokenizer
        _tokenizer = BertTokenizer.from_pretrained(PathConfig.BERT_MODEL_NAME)

        # 推荐模型
        _model = BERTMultiClassifier(
            num_classes=len(_scenic_names),
            bert_model_name=PathConfig.BERT_MODEL_NAME
        ).to(device)
        if os.path.exists(PathConfig.RECOMMEND_MODEL_PATH):
            _model.load_state_dict(
                torch.load(PathConfig.RECOMMEND_MODEL_PATH, map_location=device)
            )
            print("[recommend_service] 推荐模型加载成功")
        _model.eval()

        # 景区信息
        _scenic_info = _load_scenic_info()
        _scenic_keywords = _load_scenic_keywords()

        _loaded = True
        print(f"[recommend_service] 加载完成: {len(_scenic_names)}个景区")
        return _tokenizer, _model, _scenic_names, _scenic_info, _scenic_keywords


def recommend(text, top_k=3):
    """推荐景区，返回结果列表"""
    tokenizer, model, scenic_names, scenic_info, scenic_keywords = load_model()
    device = _get_device()

    encoding = tokenizer(
        text, truncation=True, padding='max_length',
        max_length=DataConfig.MAX_SEQ_LEN, return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probs, k=min(top_k, len(scenic_names)), dim=1)

    results = []
    for i in range(min(top_k, len(scenic_names))):
        idx = top_indices[0][i].item()
        prob = round(top_probs[0][i].item(), 4)
        name = scenic_names[idx]
        info = scenic_info.get(name, {})
        kw = scenic_keywords.get(name, {})

        results.append({
            'rank': i + 1,
            'scenic': name,
            'confidence': prob,
            'address': info.get('address', ''),
            'price': info.get('price', ''),
            'reason': info.get('reason', ''),
            'image': info.get('image', ''),
            'pos_keywords': kw.get('pos_keywords', ''),
            'neg_keywords': kw.get('neg_keywords', ''),
        })
    return results


def get_scenic_info(name):
    """获取单个景区详情"""
    _, _, scenic_names, scenic_info, scenic_keywords = load_model()
    info = scenic_info.get(name, {})
    kw = scenic_keywords.get(name, {})
    return {
        'name': name,
        'address': info.get('address', ''),
        'price': info.get('price', ''),
        'reason': info.get('reason', ''),
        'intro': info.get('intro', ''),
        'hours': info.get('hours', ''),
        'policy': info.get('policy', ''),
        'phone': info.get('phone', ''),
        'image': info.get('image', ''),
        'image2': info.get('image2', ''),
        'pos_keywords': kw.get('pos_keywords', ''),
        'neg_keywords': kw.get('neg_keywords', ''),
    }


def get_all_scenics():
    """获取所有景区列表"""
    _, _, scenic_names, scenic_info, _ = load_model()
    return [
        {'name': name, **scenic_info.get(name, {})}
        for name in scenic_names
    ]