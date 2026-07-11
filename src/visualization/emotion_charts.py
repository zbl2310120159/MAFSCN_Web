"""
情感分布可视化模块：饼图、柱状图、正面率排名图
从analysis_service获取统计数据
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import PathConfig

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_pie_charts():
    """为每个景区生成情感分布饼图"""
    from src.services.analysis_service import get_sentiment_ratio

    output_dir = os.path.join(PathConfig.FIGURE_DIR, 'emotion', 'pie')
    os.makedirs(output_dir, exist_ok=True)

    data = get_sentiment_ratio()
    if not data:
        print("  ⚠️ 无情感占比数据")
        return

    colors = ['#ef4444', '#9ca3af', '#22c55e']  # 负面、中性、正面
    labels = ['负面', '中性', '正面']

    for item in data:
        name = item['name']
        counts = [item['negative_count'], item['neutral_count'], item['positive_count']]
        total = sum(counts)
        if total == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 8))
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 12}
        )
        for autotext in autotexts:
            autotext.set_fontsize(11)
        ax.set_title(f'{name} - 情感分布', fontsize=16)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ 饼图已保存: {output_dir}")


def plot_bar_charts():
    """生成所有景区的情感分布柱状图"""
    from src.services.analysis_service import get_scenic_stats

    output_dir = os.path.join(PathConfig.FIGURE_DIR, 'emotion', 'bar')
    os.makedirs(output_dir, exist_ok=True)

    data = get_scenic_stats()
    if not data:
        print("  ⚠️ 无统计数据")
        return

    # 按正面评论数排序
    data_sorted = sorted(data, key=lambda x: x['positive'], reverse=True)
    names = [d['name'] for d in data_sorted]
    negatives = [d['negative'] for d in data_sorted]
    neutrals = [d['neutral'] for d in data_sorted]
    positives = [d['positive'] for d in data_sorted]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width, negatives, width, label='负面', color='#ef4444')
    ax.bar(x, neutrals, width, label='中性', color='#9ca3af')
    ax.bar(x + width, positives, width, label='正面', color='#22c55e')

    ax.set_xlabel('景区', fontsize=12)
    ax.set_ylabel('评论数量', fontsize=12)
    ax.set_title('各景区情感分布对比', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_scenic_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 柱状图已保存: {output_dir}")


def plot_positive_rate():
    """生成各景区正面率排序图"""
    from src.services.analysis_service import get_sentiment_ratio

    output_dir = os.path.join(PathConfig.FIGURE_DIR, 'emotion', 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    data = get_sentiment_ratio()
    if not data:
        print("  ⚠️ 无情感占比数据")
        return

    # 按正面率排序
    data_sorted = sorted(data, key=lambda x: x['positive_pct'])
    names = [d['name'] for d in data_sorted]
    rates = [d['positive_pct'] / 100 for d in data_sorted]

    # 颜色编码：>80%绿，>70%黄，<70%红
    colors = ['#22c55e' if r > 0.8 else '#f59e0b' if r > 0.7 else '#ef4444' for r in rates]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(names, rates, color=colors)
    ax.set_xlabel('正面率', fontsize=12)
    ax.set_title('各景区正面率排名', fontsize=16)
    ax.set_xlim(0, 1.0)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{rate:.1%}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'positive_rate_ranking.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ 正面率排名图已保存: {output_dir}")


def generate_all_emotion_charts():
    """生成所有情感分布图表"""
    print("📊 生成情感分布图表...")
    plot_pie_charts()
    plot_bar_charts()
    plot_positive_rate()
    print("✅ 情感分布图表生成完成！")


if __name__ == '__main__':
    generate_all_emotion_charts()