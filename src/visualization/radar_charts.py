"""
景区画像雷达图模块
从analysis_service获取维度评分数据
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


def plot_radar_charts():
    """绘制各景区画像雷达图"""
    from src.services.analysis_service import get_scenic_comparison

    output_dir = os.path.join(PathConfig.FIGURE_DIR, 'radar')
    os.makedirs(output_dir, exist_ok=True)

    data = get_scenic_comparison()
    if not data:
        print("  ⚠️ 无对比数据")
        return

    # 获取维度名称
    first_dims = data[0].get('dimension_scores', {})
    dimensions = list(first_dims.keys())
    if not dimensions:
        print("  ⚠️ 无维度评分数据")
        return

    N = len(dimensions)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 配色
    colors = ['#ef4444', '#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6',
              '#06b6d4', '#ec4899', '#14b8a6', '#f97316', '#6366f1',
              '#84cc16', '#e11d48']

    # ====== 1. 所有景区叠加雷达图 ======
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    for i, item in enumerate(data):
        ds = item.get('dimension_scores', {})
        # 转为0-10分制
        values = [round(ds.get(dim, 0) * 10, 1) for dim in dimensions]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=item['name'],
                color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.05, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_title('各景区画像雷达图', fontsize=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_scenic_radar.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 综合雷达图已保存: {output_dir}")

    # ====== 2. 单独景区雷达图 ======
    single_dir = os.path.join(output_dir, 'single')
    os.makedirs(single_dir, exist_ok=True)

    for i, item in enumerate(data):
        ds = item.get('dimension_scores', {})
        values = [round(ds.get(dim, 0) * 10, 1) for dim in dimensions]
        values += values[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, values, 'o-', linewidth=2.5,
                color=colors[i % len(colors)], label=item['name'])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_title(f'{item["name"]} - 维度评分', fontsize=16, pad=20)

        # 标注分数
        for j, (angle, val) in enumerate(zip(angles[:-1], values[:-1])):
            ax.annotate(f'{val}', xy=(angle, val), fontsize=10,
                        ha='center', va='bottom', color=colors[i % len(colors)])

        plt.tight_layout()
        plt.savefig(os.path.join(single_dir, f'{item["name"]}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ 单独雷达图已保存: {single_dir}")

    # ====== 3. Top3对比雷达图 ======
    top3 = data[:3]  # 按avg_score排序的前3
    if len(top3) >= 2:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        for i, item in enumerate(top3):
            ds = item.get('dimension_scores', {})
            values = [round(ds.get(dim, 0) * 10, 1) for dim in dimensions]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2.5, label=item['name'],
                    color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions, fontsize=12)
        ax.set_ylim(0, 10)
        ax.set_title('Top3 景区维度对比', fontsize=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=12)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top3_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Top3对比雷达图已保存: {output_dir}")


def generate_all_radar_charts():
    """生成所有雷达图"""
    print("📊 生成景区画像雷达图...")
    plot_radar_charts()
    print("✅ 雷达图生成完成！")


if __name__ == '__main__':
    generate_all_radar_charts()