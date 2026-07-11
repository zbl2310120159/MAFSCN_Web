"""
数据预处理脚本
功能：合并全量数据+差评数据，列筛选，去重，文本清洗，分词，标签生成
"""

import pandas as pd
import jieba
import re
import os
import sys

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')

# 保留的核心列
KEEP_COLUMNS = ['评论ID', '用户昵称', '综合评分', '评论内容', '发布时间',
                '有用数', '回复数', '图片数量', 'IP归属地', '是否有视频']

# 情感标签映射：1-2→负面(0), 3→中性(1), 4-5→正面(2)
SENTIMENT_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

# 停用词
STOPWORDS = {
    '的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '也', '很', '到', '说', '去', '会', '这', '那', '要', '可以', '我们', '他们',
    '你们', '自己', '这个', '那个', '什么', '怎么', '为什么', '因为', '所以', '但是',
    '如果', '虽然', '然而', '而且', '或者', '就是', '只是', '还是', '又', '才',
    '哦', '吧', '呢', '啦', '嘛', '呀', '啊', '嗯', '哈哈', '呵呵', '嘻嘻',
    '非常', '比较', '特别', '有点', '一些', '一点', '一下', '一直', '已经',
    '应该', '需要', '想要', '觉得', '感觉', '知道', '希望', '建议',
    '还有', '只有', '只要', '由于', '于是', '因而', '因此',
    '从而', '并且', '况且', '何况', '与其', '宁可',
    '我们', '他们', '你们', '自己', '这个', '那个', '什么', '怎么', '为什么',
    '没', '没有', '不是', '被', '把', '让', '给', '从', '向', '对', '跟', '比',
    '等', '等等', '之', '其', '或', '仍', '但', '而', '与', '及', '以', '为',
    '于', '上', '下', '中', '里', '外', '前', '后', '左', '右', '来', '去',
    '过', '起', '出', '入', '回', '开', '关', '能', '得', '地', '着',
}


def clean_text(text):
    """文本清洗：去除emoji、特殊字符，保留中文、英文、数字、常用标点"""
    if pd.isna(text):
        return ""
    text = str(text)
    # 保留：中文、英文、数字、常用标点
    pattern = re.compile(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：""''（）【】、]')
    text = pattern.sub('', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def segment_text(text):
    """中文分词 + 停用词过滤"""
    if len(text) < 2:
        return ""
    words = jieba.lcut(text)
    filtered = [w.strip() for w in words if len(w.strip()) > 1 and w.strip() not in STOPWORDS]
    return ' '.join(filtered)


def extract_scenic_name(filename):
    """从文件名提取景区名：冰雪大世界差评.xlsx → 冰雪大世界"""
    name = os.path.splitext(filename)[0]
    if name.endswith('差评'):
        name = name[:-2]
    return name


def load_and_merge_data(raw_dir):
    """
    读取所有xlsx，合并并去重
    全量数据 + 差评数据，差评数据优先（去重时保留差评）
    """
    files = [f for f in os.listdir(raw_dir) if f.endswith('.xlsx')]
    
    all_dfs = []
    for f in sorted(files):
        filepath = os.path.join(raw_dir, f)
        scenic_name = extract_scenic_name(f)
        is_negative = f.endswith('差评.xlsx')
        
        df = pd.read_excel(filepath)
        df['景区'] = scenic_name
        df['数据来源'] = '差评' if is_negative else '全量'
        all_dfs.append(df)
        print(f"  读取 {f}: {len(df)}行")
    
    # 合并
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"\n合并后总行数: {len(merged)}")
    
    # 去重：同一评论ID，差评数据优先保留
    merged['_sort'] = merged['数据来源'].map({'差评': 0, '全量': 1})
    merged = merged.sort_values('_sort').drop_duplicates(subset='评论ID', keep='first')
    merged = merged.drop(columns=['_sort']).reset_index(drop=True)
    
    dup_count = len(pd.concat(all_dfs, ignore_index=True)) - len(merged)
    print(f"去重删除: {dup_count}行（差评优先保留）")
    print(f"去重后行数: {len(merged)}")
    
    return merged


def main():
    print("=" * 60)
    print("开始数据预处理")
    print("=" * 60)
    
    # 1. 读取并合并数据
    print("\n[1/6] 读取并合并数据...")
    df = load_and_merge_data(DATA_RAW)
    
    # 2. 列筛选
    print("\n[2/6] 列筛选...")
    keep_cols = KEEP_COLUMNS + ['景区', '数据来源']
    # 只保留存在的列
    actual_keep = [c for c in keep_cols if c in df.columns]
    dropped = [c for c in df.columns if c not in actual_keep]
    print(f"  保留列: {actual_keep}")
    print(f"  去除列: {dropped}")
    df = df[actual_keep]
    
    # 3. 文本清洗
    print("\n[3/6] 文本清洗...")
    df['清洗内容'] = df['评论内容'].apply(clean_text)
    empty_before = (df['清洗内容'] == '').sum()
    print(f"  清洗后空评论: {empty_before}条")
    
    # 4. 分词
    print("\n[4/6] 分词处理...")
    print("  （首次运行jieba需加载词典，稍候...）")
    df['分词结果'] = df['清洗内容'].apply(segment_text)
    
    # 5. 生成情感标签
    print("\n[5/6] 生成情感标签...")
    df['情感标签'] = df['综合评分'].map(SENTIMENT_MAP)
    unmapped = df['情感标签'].isna().sum()
    if unmapped > 0:
        print(f"  ⚠️ 未映射评分: {unmapped}条，填充为中性(1)")
        df['情感标签'] = df['情感标签'].fillna(1).astype(int)
    
    label_names = {0: '负面', 1: '中性', 2: '正面'}
    for label_val, label_name in label_names.items():
        count = (df['情感标签'] == label_val).sum()
        print(f"  {label_name}({label_val}): {count}条")
    
    # 6. 清理空分词结果
    print("\n[6/6] 清理空分词结果...")
    before = len(df)
    df = df[df['分词结果'] != ''].reset_index(drop=True)
    after = len(df)
    print(f"  删除空分词: {before - after}条")
    print(f"  最终数据: {after}条")
    
    # 统计各景区数据量
    print("\n" + "-" * 40)
    print("各景区数据量:")
    scenic_counts = df.groupby('景区').size().sort_values(ascending=False)
    for name, count in scenic_counts.items():
        neg = ((df['景区'] == name) & (df['情感标签'] == 0)).sum()
        neu = ((df['景区'] == name) & (df['情感标签'] == 1)).sum()
        pos = ((df['景区'] == name) & (df['情感标签'] == 2)).sum()
        print(f"  {name}: {count}条 (负{neg}/中{neu}/正{pos})")
    
    # 保存
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED, 'preprocessed_data.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 预处理完成，保存至: {output_path}")
    print(f"   总数据量: {len(df)}条")
    
    return df


if __name__ == '__main__':
    main()