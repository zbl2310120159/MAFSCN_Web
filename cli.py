"""
MAFSCN 统一命令行工具
用法:
  python cli.py analyze "评论内容"         - 单条评论分析(情感+推荐)
  python cli.py scenic [景区名]            - 查看景区统计/详情
  python cli.py batch <input.csv>          - 批量分析CSV评论
  python cli.py charts [--type TYPE]       - 生成离线图表
  python cli.py compare                    - 景区对比分析
"""

import os
import sys
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_analyze(args):
    """单条评论分析：情感预测 + 景区推荐"""
    text = args.text
    if not text:
        print("❌ 请输入评论内容，例如: python cli.py analyze \"景区很好玩\"")
        return

    print(f"\n{'=' * 60}")
    print(f"  📝 评论分析")
    print(f"{'=' * 60}")
    print(f"  评论内容: {text}")
    print(f"{'─' * 60}")

    # 情感分析
    try:
        from src.services.sentiment_service import predict_ensemble
        result = predict_ensemble(text)
        label = result.get('label', '未知')
        confidence = result.get('confidence', 0)
        probs = result.get('probs', [])

        sentiment_map = {0: '负面 😞', 1: '中性 😐', 2: '正面 😊'}
        print(f"\n  情感分析结果:")
        print(f"    情感倾向: {sentiment_map.get(label, label)}")
        print(f"    置信度:   {confidence:.1%}")
        if probs:
            print(f"    概率分布: 负面={probs[0]:.1%}  中性={probs[1]:.1%}  正面={probs[2]:.1%}")
    except Exception as e:
        print(f"  ⚠️ 情感分析失败: {e}")

    # 景区推荐
    try:
        from src.services.recommend_service import recommend
        recs = recommend(text, top_k=args.top_k)
        if recs:
            print(f"\n  🏔️ 推荐景区 (Top {len(recs)}):")
            for i, rec in enumerate(recs, 1):
                name = rec.get('name', rec.get('景区', '未知'))
                score = rec.get('score', rec.get('similarity', 0))
                print(f"    {i}. {name}  (相似度: {score:.3f})")
    except Exception as e:
        print(f"  ⚠️ 景区推荐失败: {e}")

    print(f"\n{'=' * 60}")


def cmd_scenic(args):
    """查看景区统计信息"""
    from src.services.analysis_service import get_scenic_stats, get_scenic_comparison

    name = args.name

    print(f"\n{'=' * 60}")
    print(f"  🏔️ 景区信息")
    print(f"{'=' * 60}")

    # 获取统计
    stats = get_scenic_stats()

    if name:
        # 显示指定景区详情
        target = None
        for s in stats:
            if name in s.get('name', ''):
                target = s
                break

        if not target:
            print(f"  ❌ 未找到景区: {name}")
            print(f"  可用景区: {', '.join(s['name'] for s in stats)}")
            return

        print(f"\n  景区: {target['name']}")
        print(f"  {'─' * 40}")
        print(f"  总评论数:   {target.get('total', 0)}")
        print(f"  正面评论:   {target.get('positive', 0)} ({target.get('positive_pct', 0):.1f}%)")
        print(f"  中性评论:   {target.get('neutral', 0)} ({target.get('neutral_pct', 0):.1f}%)")
        print(f"  负面评论:   {target.get('negative', 0)} ({target.get('negative_pct', 0):.1f}%)")
        print(f"  平均评分:   {target.get('avg_score', 0):.2f}")

        # 维度评分
        try:
            from src.services.analysis_service import get_dimension_scores
            dims = get_dimension_scores(name)
            if dims and 'dimensions' in dims:
                print(f"\n  维度评分:")
                for dim, score in dims['dimensions'].items():
                    bar = '█' * int(score * 10) + '░' * (10 - int(score * 10))
                    print(f"    {dim:6s} [{bar}] {score * 10:.1f}/10")
        except Exception as e:
            print(f"  ⚠️ 维度评分获取失败: {e}")

    else:
        # 显示所有景区概览
        print(f"\n  {'景区':12s} {'评论数':>6s} {'正面率':>7s} {'负面率':>7s} {'均分':>5s}")
        print(f"  {'─' * 42}")
        for s in stats:
            print(f"  {s['name']:12s} {s.get('total',0):>6d} "
                  f"{s.get('positive_pct',0):>6.1f}% "
                  f"{s.get('negative_pct',0):>6.1f}% "
                  f"{s.get('avg_score',0):>5.2f}")

    print(f"\n{'=' * 60}")


def cmd_batch(args):
    """批量分析CSV评论文件"""
    import pandas as pd

    input_file = args.input
    if not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        return

    print(f"\n{'=' * 60}")
    print(f"  📊 批量评论分析")
    print(f"{'=' * 60}")
    print(f"  输入文件: {input_file}")

    # 读取CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    # 查找评论列
    comment_col = None
    for col in ['评论内容', 'comment', 'text', 'content', '评论']:
        if col in df.columns:
            comment_col = col
            break

    if not comment_col:
        print(f"❌ 未找到评论列，可用列: {list(df.columns)}")
        return

    texts = df[comment_col].dropna().tolist()
    print(f"  评论数量: {len(texts)}")
    print(f"  评论列:   {comment_col}")
    print(f"{'─' * 60}")

    # 批量情感分析
    try:
        from src.services.sentiment_service import analyze_batch
        results = analyze_batch(texts[:args.limit] if args.limit else texts)

        # 统计
        pos = sum(1 for r in results if r.get('label') == 2)
        neu = sum(1 for r in results if r.get('label') == 1)
        neg = sum(1 for r in results if r.get('label') == 0)

        print(f"\n  分析结果:")
        print(f"    总数: {len(results)}")
        print(f"    正面: {pos} ({pos/len(results):.1%})")
        print(f"    中性: {neu} ({neu/len(results):.1%})")
        print(f"    负面: {neg} ({neg/len(results):.1%})")

        # 保存结果
        if args.output:
            result_df = df.copy()
            sentiments = []
            confidences = []
            for r in results:
                sentiments.append(['负面', '中性', '正面'][r.get('label', 1)])
                confidences.append(f"{r.get('confidence', 0):.2%}")

            # 对齐行数
            while len(sentiments) < len(result_df):
                sentiments.append('')
                confidences.append('')

            result_df['情感标签'] = sentiments[:len(result_df)]
            result_df['置信度'] = confidences[:len(result_df)]
            result_df.to_csv(args.output, index=False, encoding='utf-8-sig')
            print(f"\n  ✅ 结果已保存: {args.output}")

    except Exception as e:
        print(f"❌ 批量分析失败: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'=' * 60}")


def cmd_charts(args):
    """生成离线图表"""
    from scripts.generate_charts import main as gen_main
    # 复用generate_charts的逻辑
    sys.argv = ['generate_charts.py', '--type', args.type]
    if args.output:
        sys.argv += ['--output', args.output]
    gen_main()


def cmd_compare(args):
    """景区对比分析"""
    from src.services.analysis_service import get_scenic_comparison

    print(f"\n{'=' * 60}")
    print(f"  📊 景区对比分析")
    print(f"{'=' * 60}")

    data = get_scenic_comparison()
    if not data:
        print("  ❌ 无对比数据")
        return

    # 表格输出
    dims = list(data[0].get('dimension_scores', {}).keys()) if data else []

    header = f"  {'景区':12s} {'均分':>5s} {'评论':>5s} {'好评率':>6s}"
    for d in dims:
        header += f" {d:>5s}"
    print(header)
    print(f"  {'─' * (len(header) - 2)}")

    for item in data:
        row = f"  {item['name']:12s} {item.get('avg_score',0):>5.2f} " \
              f"{item.get('total',0):>5d} {item.get('positive_pct',0):>5.1f}%"
        ds = item.get('dimension_scores', {})
        for d in dims:
            row += f" {ds.get(d, 0) * 10:>5.1f}"
        print(row)

    # 排名
    print(f"\n  🏆 综合评分排名:")
    sorted_data = sorted(data, key=lambda x: x.get('avg_score', 0), reverse=True)
    for i, item in enumerate(sorted_data, 1):
        medal = ['🥇', '🥈', '🥉'][i-1] if i <= 3 else f" {i}."
        print(f"    {medal} {item['name']:12s} {item.get('avg_score',0):.2f}")

    print(f"\n{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='MAFSCN 多维度景区评论分析系统 - 命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cli.py analyze "故宫非常壮观，值得一去"
  python cli.py scenic 故宫
  python cli.py scenic
  python cli.py batch data/comments.csv --output results.csv
  python cli.py charts --type wordcloud
  python cli.py compare
        """)

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # analyze 子命令
    p_analyze = subparsers.add_parser('analyze', help='单条评论分析')
    p_analyze.add_argument('text', nargs='?', help='评论内容')
    p_analyze.add_argument('--top-k', type=int, default=3, help='推荐景区数量(默认3)')

    # scenic 子命令
    p_scenic = subparsers.add_parser('scenic', help='查看景区信息')
    p_scenic.add_argument('name', nargs='?', help='景区名称(留空显示全部)')

    # batch 子命令
    p_batch = subparsers.add_parser('batch', help='批量分析CSV评论')
    p_batch.add_argument('input', help='输入CSV文件路径')
    p_batch.add_argument('--output', '-o', help='输出CSV文件路径')
    p_batch.add_argument('--limit', type=int, help='限制分析数量')

    # charts 子命令
    p_charts = subparsers.add_parser('charts', help='生成离线图表')
    p_charts.add_argument('--type', '-t', default='all',
                          choices=['all', 'wordcloud', 'emotion', 'radar'],
                          help='图表类型(默认all)')
    p_charts.add_argument('--output', '-o', help='输出目录')

    # compare 子命令
    subparsers.add_parser('compare', help='景区对比分析')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        'analyze': cmd_analyze,
        'scenic': cmd_scenic,
        'batch': cmd_batch,
        'charts': cmd_charts,
        'compare': cmd_compare,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()