"""
词云图生成模块
从analysis_service获取词频数据，生成正面/负面词云图
"""

import os
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')  # 无GUI环境
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from config import PathConfig

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 尝试查找中文字体
_FONT_PATH = None
for fp in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/simhei.ttf',
           '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']:
    if os.path.exists(fp):
        _FONT_PATH = fp
        break


def generate_wordcloud(word_freq, output_path, title=None, max_words=100,
                       colormap='viridis', width=800, height=600):
    """生成词云图

    Args:
        word_freq: [{name: str, value: int}, ...] 词频列表
        output_path: 输出路径
        title: 图表标题
        max_words: 最大词数
        colormap: matplotlib colormap名
        width: 图片宽度
        height: 图片高度
    """
    if not word_freq:
        print(f"  ⚠️ 无词频数据，跳过: {output_path}")
        return

    # 转为 {word: freq} 字典
    freq_dict = {item['name']: item['value'] for item in word_freq}

    wc_kwargs = dict(
        width=width, height=height,
        max_words=max_words,
        background_color='white',
        colormap=colormap,
    )
    if _FONT_PATH:
        wc_kwargs['font_path'] = _FONT_PATH

    wc = WordCloud(**wc_kwargs)
    wc.generate_from_frequencies(freq_dict)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=16)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_all_wordclouds():
    """生成所有景区的正面/负面词云图"""
    from src.services.analysis_service import get_sentiment_wordcloud_data, get_scenic_stats

    output_dir = os.path.join(PathConfig.FIGURE_DIR, 'wordcloud')
    os.makedirs(output_dir, exist_ok=True)

    scenics = [s['name'] for s in get_scenic_stats()]
    print(f"📊 生成词云图: {len(scenics)} 个景区")

    for scenic in scenics:
        print(f"  🔄 {scenic}...")
        data = get_sentiment_wordcloud_data(scenic)

        # 正面词云
        pos_words = data.get('positive', [])
        if pos_words:
            generate_wordcloud(
                pos_words,
                os.path.join(output_dir, f'{scenic}_正面.png'),
                title=f'{scenic} - 正面评论词云',
                colormap='YlGn'
            )

        # 负面词云
        neg_words = data.get('negative', [])
        if neg_words:
            generate_wordcloud(
                neg_words,
                os.path.join(output_dir, f'{scenic}_负面.png'),
                title=f'{scenic} - 负面评论词云',
                colormap='OrRd'
            )

    # 生成整体词云
    print("  🔄 整体词云...")
    all_pos, all_neg = [], []
    for scenic in scenics:
        data = get_sentiment_wordcloud_data(scenic)
        all_pos.extend(data.get('positive', []))
        all_neg.extend(data.get('negative', []))

    # 合并同词频
    def _merge(words):
        freq = {}
        for w in words:
            freq[w['name']] = freq.get(w['name'], 0) + w['value']
        return [{'name': k, 'value': v} for k, v in sorted(freq.items(), key=lambda x: -x[1])]

    generate_wordcloud(_merge(all_pos)[:100], os.path.join(output_dir, '整体_正面.png'),
                       '所有景区 - 正面评论词云', colormap='YlGn')
    generate_wordcloud(_merge(all_neg)[:100], os.path.join(output_dir, '整体_负面.png'),
                       '所有景区 - 负面评论词云', colormap='OrRd')

    print(f"✅ 词云图已保存: {output_dir}")


if __name__ == '__main__':
    generate_all_wordclouds()