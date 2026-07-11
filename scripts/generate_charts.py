"""
一键生成所有离线图表
运行: python scripts/generate_charts.py [--type TYPE]

type选项:
  all       - 生成全部图表(默认)
  wordcloud - 仅词云图
  emotion   - 仅情感分布图
  radar     - 仅雷达图
"""

import os
import sys
import argparse
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PathConfig


def generate_wordcloud_charts():
    """生成词云图"""
    from src.visualization.wordcloud_charts import generate_all_wordclouds
    generate_all_wordclouds()


def generate_emotion_charts():
    """生成情感分布图"""
    from src.visualization.emotion_charts import generate_all_emotion_charts
    generate_all_emotion_charts()


def generate_radar_charts():
    """生成雷达图"""
    from src.visualization.radar_charts import generate_all_radar_charts
    generate_all_radar_charts()


def main():
    parser = argparse.ArgumentParser(description='MAFSCN 离线图表生成工具')
    parser.add_argument('--type', '-t', default='all',
                        choices=['all', 'wordcloud', 'emotion', 'radar'],
                        help='图表类型: all/wordcloud/emotion/radar (默认: all)')
    parser.add_argument('--output', '-o', default=None,
                        help='输出目录 (默认: results/figures/)')
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(PathConfig.FIGURE_DIR, exist_ok=True)

    print("=" * 60)
    print("  MAFSCN 离线图表生成工具")
    print("=" * 60)
    print(f"  输出目录: {PathConfig.FIGURE_DIR}")
    print(f"  图表类型: {args.type}")
    print("=" * 60)

    start_time = time.time()

    generators = {
        'wordcloud': ('词云图', generate_wordcloud_charts),
        'emotion': ('情感分布图', generate_emotion_charts),
        'radar': ('雷达图', generate_radar_charts),
    }

    if args.type == 'all':
        tasks = list(generators.items())
    else:
        tasks = [(args.type, generators[args.type])]

    for chart_type, (name, func) in tasks:
        print(f"\n{'─' * 40}")
        print(f"📊 正在生成: {name}")
        print(f"{'─' * 40}")
        try:
            func()
        except Exception as e:
            print(f"❌ {name}生成失败: {e}")
            import traceback
            traceback.print_exc()

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  ✅ 全部完成！耗时 {elapsed:.1f} 秒")
    print(f"  📁 图表保存位置: {PathConfig.FIGURE_DIR}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()