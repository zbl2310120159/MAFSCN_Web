"""
数据分析统计服务
基于preprocessed_data.csv，返回可JSON序列化数据
"""

import threading
from collections import Counter

import pandas as pd
import numpy as np
import jieba

from config import PathConfig, DataConfig

# 全局缓存
_df = None
_lock = threading.Lock()

# 列名兼容
_COL_SCENIC = None  # '景区' 或 '来源景区'
_COL_LABEL = None   # '情感标签' 或 'label'
_COL_SEGMENT = None # '分词结果' 或 'segmented'

# 停用词
_STOPWORDS = set(
    '的 了 是 在 和 有 我 这 也 不 都 就 很 还 一个 非常 真的 感觉 地方 景区 景点 '
    '值得 不错 体验 讲解 导游 很多 好玩 推荐 没有 不要 可以 就是 还是 一次 时间 '
    '去 去 去 到 会 能 说 那 什么 怎么 但 而 却 又 又 被 把 让 给 从 向 对 与 '
    '为 因为 所以 如果 虽然 但是 而且 或者 以及 之 其 等 个 各 该 本 此 另 外 '
    '多 少 大 小 长 短 高 低 好 坏 新 旧 前 后 左 右 上 下 中 内 外'.split()
)

# 维度关键词
_DIM_KEYWORDS = {
    '风景': ['风景', '景色', '景观', '自然', '天池', '雪山', '云海', '冰雕', '壮观', '美丽', '漂亮', '震撼'],
    '服务': ['服务', '态度', '热情', '专业', '导游', '讲解', '工作人员', '志愿者', '贴心'],
    '价格': ['价格', '门票', '性价比', '便宜', '贵', '划算', '超值', '收费', '免费', '优惠'],
    '设施': ['设施', '项目', '索道', '滑梯', '电梯', '厕所', '交通', '停车', '缆车', '观光车'],
    '体验': ['好玩', '有趣', '震撼', '值得', '推荐', '开心', '难忘', '满意', '惊喜', '失望'],
}


def _get_df():
    """获取缓存DataFrame，兼容列名"""
    global _df, _COL_SCENIC, _COL_LABEL, _COL_SEGMENT
    if _df is not None:
        return _df

    with _lock:
        if _df is not None:
            return _df
        print("[analysis_service] 正在加载数据...")
        _df = pd.read_csv(PathConfig.DATA_PREPROCESSED)

        # 兼容列名
        _COL_SCENIC = '景区' if '景区' in _df.columns else '来源景区'
        _COL_LABEL = '情感标签' if '情感标签' in _df.columns else 'label'
        _COL_SEGMENT = '分词结果' if '分词结果' in _df.columns else 'segmented'

        # 标准化情感标签为int
        _df[_COL_LABEL] = pd.to_numeric(_df[_COL_LABEL], errors='coerce').fillna(1).astype(int)

        print(f"[analysis_service] 数据加载完成: {len(_df)}条, 列名: 景区={_COL_SCENIC}, 标签={_COL_LABEL}")
        return _df


def _label_name(val):
    """标签值→名称"""
    return {0: '负面', 1: '中性', 2: '正面'}.get(int(val), '中性')


def _scenic_df(name):
    """获取某景区子DataFrame"""
    df = _get_df()
    return df[df[_COL_SCENIC] == name]


# ============================================================
# 1. 各景区评论数量统计
# ============================================================
def get_scenic_stats():
    """返回 [{name, total, positive, neutral, negative, positive_pct, neutral_pct, negative_pct, avg_score}]"""
    df = _get_df()
    score_col = '综合评分'
    results = []
    for name in df[_COL_SCENIC].dropna().unique():
        sdf = df[df[_COL_SCENIC] == name]
        total = len(sdf)
        pos = int((sdf[_COL_LABEL] == 2).sum())
        neu = int((sdf[_COL_LABEL] == 1).sum())
        neg = int((sdf[_COL_LABEL] == 0).sum())
        avg_score = float(sdf[score_col].mean()) if score_col in sdf.columns else 0.0
        results.append({
            'name': name, 'total': total,
            'positive': pos, 'neutral': neu, 'negative': neg,
            'positive_pct': round(pos / total * 100, 1) if total > 0 else 0,
            'neutral_pct': round(neu / total * 100, 1) if total > 0 else 0,
            'negative_pct': round(neg / total * 100, 1) if total > 0 else 0,
            'avg_score': round(avg_score, 2),
        })
    results.sort(key=lambda x: x['total'], reverse=True)
    return results


# ============================================================
# 2. 评分分布
# ============================================================
def get_score_distribution():
    """返回 {overall:[{score,count}], by_scenic:[{name,scores:[{score,count}]}]}"""
    df = _get_df()
    score_col = '综合评分'
    if score_col not in df.columns:
        return {'overall': [], 'by_scenic': []}

    # 总体分布
    overall = []
    for s in range(1, 6):
        overall.append({'score': s, 'count': int((df[score_col] == s).sum())})

    # 按景区分组
    by_scenic = []
    for name in df[_COL_SCENIC].dropna().unique():
        sdf = df[df[_COL_SCENIC] == name]
        scores = [{'score': s, 'count': int((sdf[score_col] == s).sum())} for s in range(1, 6)]
        by_scenic.append({'name': name, 'scores': scores})

    return {'overall': overall, 'by_scenic': by_scenic}


# ============================================================
# 3. 各景区正/负/中性占比
# ============================================================
def get_sentiment_ratio():
    """返回 [{name, positive_pct, neutral_pct, negative_pct, positive_count, neutral_count, negative_count}]"""
    df = _get_df()
    results = []
    for name in df[_COL_SCENIC].dropna().unique():
        sdf = df[df[_COL_SCENIC] == name]
        total = len(sdf)
        if total == 0:
            continue
        pos = int((sdf[_COL_LABEL] == 2).sum())
        neu = int((sdf[_COL_LABEL] == 1).sum())
        neg = int((sdf[_COL_LABEL] == 0).sum())
        results.append({
            'name': name,
            'positive_pct': round(pos / total * 100, 1),
            'neutral_pct': round(neu / total * 100, 1),
            'negative_pct': round(neg / total * 100, 1),
            'positive_count': pos, 'neutral_count': neu, 'negative_count': neg,
        })
    return results


# ============================================================
# 4. 评论时间趋势（按月）
# ============================================================
def get_time_trend():
    """返回 {overall:[{month,total,positive,neutral,negative}], by_scenic:[...]}"""
    df = _get_df()
    time_col = '发布时间'
    if time_col not in df.columns:
        return {'overall': [], 'by_scenic': []}

    df_copy = df.copy()
    df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
    df_copy = df_copy.dropna(subset=[time_col])
    df_copy['month'] = df_copy[time_col].dt.to_period('M').astype(str)

    def _calc_trend(sub_df):
        trend = []
        for month in sorted(sub_df['month'].unique()):
            mdf = sub_df[sub_df['month'] == month]
            trend.append({
                'month': month, 'total': len(mdf),
                'positive': int((mdf[_COL_LABEL] == 2).sum()),
                'neutral': int((mdf[_COL_LABEL] == 1).sum()),
                'negative': int((mdf[_COL_LABEL] == 0).sum()),
            })
        return trend

    overall = _calc_trend(df_copy)
    by_scenic = []
    for name in df_copy[_COL_SCENIC].dropna().unique():
        sdf = df_copy[df_copy[_COL_SCENIC] == name]
        by_scenic.append({'name': name, 'trend': _calc_trend(sdf)})

    return {'overall': overall, 'by_scenic': by_scenic}


# ============================================================
# 5. 正面/负面关键词TOP10
# ============================================================
def get_keywords(scenic, sentiment):
    """
    scenic: 景区名称
    sentiment: '正面'/'负面'/'中性'
    返回 [{word, count}]
    """
    df = _get_df()
    label_val = {'负面': 0, '中性': 1, '正面': 2}.get(sentiment, 1)
    sdf = df[(df[_COL_SCENIC] == scenic) & (df[_COL_LABEL] == label_val)]

    if len(sdf) == 0:
        return []

    # 优先使用分词结果列
    if _COL_SEGMENT in sdf.columns:
        texts = sdf[_COL_SEGMENT].dropna().astype(str).tolist()
        all_words = []
        for t in texts:
            words = t.split()
            all_words.extend(w for w in words if w not in _STOPWORDS and len(w) >= 2)
    else:
        # 回退到jieba分词
        texts = sdf['评论内容'].dropna().astype(str).tolist()
        all_words = []
        for t in texts:
            words = jieba.lcut(t)
            all_words.extend(w for w in words if w not in _STOPWORDS and len(w) >= 2)

    counter = Counter(all_words)
    return [{'word': w, 'count': c} for w, c in counter.most_common(10)]


# ============================================================
# 6. 景区多维评分
# ============================================================
def get_dimension_scores(scenic):
    """返回 {name, dimensions:{风景:score, 服务:score, 价格:score, 设施:score, 体验:score}}"""
    df = _get_df()
    sdf = df[df[_COL_SCENIC] == scenic]
    if len(sdf) == 0:
        return {'name': scenic, 'dimensions': {}}

    texts = sdf['评论内容'].dropna().astype(str).tolist()
    n = len(texts)
    if n == 0:
        return {'name': scenic, 'dimensions': {}}

    # 按评论数归一化：统计提及各维度关键词的评论占比
    scores = {}
    for dim, words in _DIM_KEYWORDS.items():
        # 统计提及该维度关键词的评论数（非总提及次数）
        mention_count = sum(1 for t in texts if any(w in t for w in words))
        scores[dim] = round(min(mention_count / n, 1.0), 2)

    return {'name': scenic, 'dimensions': scores}


# ============================================================
# 7. 词云数据
# ============================================================
def get_sentiment_wordcloud_data(scenic):
    """返回 {positive:[{name,value}], negative:[{name,value}]}"""
    df = _get_df()
    sdf = df[df[_COL_SCENIC] == scenic]

    def _extract_words(sub_df, top_n=50):
        if len(sub_df) == 0:
            return []
        if _COL_SEGMENT in sub_df.columns:
            texts = sub_df[_COL_SEGMENT].dropna().astype(str).tolist()
            all_words = []
            for t in texts:
                words = t.split()
                all_words.extend(w for w in words if w not in _STOPWORDS and len(w) >= 2)
        else:
            texts = sub_df['评论内容'].dropna().astype(str).tolist()
            all_words = []
            for t in texts:
                words = jieba.lcut(t)
                all_words.extend(w for w in words if w not in _STOPWORDS and len(w) >= 2)
        counter = Counter(all_words)
        return [{'name': w, 'value': c} for w, c in counter.most_common(top_n)]

    pos_df = sdf[sdf[_COL_LABEL] == 2]
    neg_df = sdf[sdf[_COL_LABEL] == 0]

    return {
        'positive': _extract_words(pos_df, 50),
        'negative': _extract_words(neg_df, 50),
    }


# ============================================================
# 8. 评论长度分布
# ============================================================
def get_comment_length_distribution():
    """返回 [{range, count}]"""
    df = _get_df()
    content_col = '评论内容'
    if content_col not in df.columns:
        return []

    lengths = df[content_col].dropna().astype(str).str.len()
    bins = [(0, 20), (20, 50), (50, 100), (100, 200), (200, float('inf'))]
    labels = ['0-20', '20-50', '50-100', '100-200', '200+']

    results = []
    for (lo, hi), label in zip(bins, labels):
        if hi == float('inf'):
            count = int((lengths >= lo).sum())
        else:
            count = int(((lengths >= lo) & (lengths < hi)).sum())
        results.append({'range': label, 'count': count})

    return results


# ============================================================
# 9. 最有用评论
# ============================================================
def get_useful_comments(scenic, top_k=5):
    """返回 [{content, score, useful_count, sentiment, time}]"""
    df = _get_df()
    sdf = df[df[_COL_SCENIC] == scenic].copy()

    useful_col = '有用数'
    if useful_col not in sdf.columns:
        return []

    sdf[useful_col] = pd.to_numeric(sdf[useful_col], errors='coerce').fillna(0)
    sdf = sdf.sort_values(useful_col, ascending=False).head(top_k)

    results = []
    for _, row in sdf.iterrows():
        content = str(row.get('评论内容', ''))
        score = int(row.get('综合评分', 0)) if pd.notna(row.get('综合评分')) else 0
        useful = int(row.get('有用数', 0)) if pd.notna(row.get('有用数')) else 0
        label = int(row.get(_COL_LABEL, 1)) if pd.notna(row.get(_COL_LABEL)) else 1
        time_val = str(row.get('发布时间', '')) if pd.notna(row.get('发布时间')) else ''
        results.append({
            'content': content, 'score': score,
            'useful_count': useful, 'sentiment': _label_name(label),
            'time': time_val,
        })
    return results


# ============================================================
# 10. 景区综合对比
# ============================================================
def get_scenic_comparison():
    """返回 [{name, total, avg_score, positive_pct, dimension_scores}]"""
    df = _get_df()
    score_col = '综合评分'
    results = []

    for name in df[_COL_SCENIC].dropna().unique():
        sdf = df[df[_COL_SCENIC] == name]
        total = len(sdf)
        if total == 0:
            continue

        pos = int((sdf[_COL_LABEL] == 2).sum())
        pos_pct = round(pos / total * 100, 1)

        avg_score = 0.0
        if score_col in sdf.columns:
            avg_score = round(float(sdf[score_col].mean()), 2)

        # 维度评分
        texts = sdf['评论内容'].dropna().astype(str).tolist()
        n = len(texts)
        dim_scores = {}
        if n > 0:
            for dim, words in _DIM_KEYWORDS.items():
                mention_count = sum(1 for t in texts if any(w in t for w in words))
                dim_scores[dim] = round(min(mention_count / n, 1.0), 2)

        results.append({
            'name': name, 'total': total,
            'avg_score': avg_score, 'positive_pct': pos_pct,
            'dimension_scores': dim_scores,
        })

    results.sort(key=lambda x: x['avg_score'], reverse=True)
    return results