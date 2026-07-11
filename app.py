"""
MAFSCN 景区智能分析系统 - Flask 统一入口
游客端 + 管理端所有路由
"""

import os
import sys
import io
import csv
import uuid
import socket
import threading
import warnings
from datetime import datetime

# ============================================================
# Windows 中文编码兼容补丁（必须在所有 socket/IO 操作之前）
# ============================================================
if sys.platform == 'win32':
    # 修复 GBK 终端输出乱码
    if sys.stdout and hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, 'buffer'):
        try:
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass
    # 修复中文主机名导致 socket.getfqdn 报错
    original_getfqdn = socket.getfqdn
    try:
        original_getfqdn('127.0.0.1')
    except UnicodeDecodeError:
        socket.getfqdn = lambda name: 'localhost'

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

import torch
import numpy as np
import pandas as pd
from flask import (
    Flask, request, jsonify, send_file,
    send_from_directory, render_template
)

# ============================================================
# 项目路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# 导入配置
from config import PathConfig, DataConfig, ModelConfig, FlaskConfig, create_dirs

# 导入服务层
from src.services.sentiment_service import load_models, predict_ensemble, analyze_batch
from src.services.recommend_service import load_model, recommend, get_scenic_info, get_all_scenics
from src.services.analysis_service import (
    get_scenic_stats, get_score_distribution, get_sentiment_ratio,
    get_time_trend, get_keywords, get_dimension_scores,
    get_sentiment_wordcloud_data, get_comment_length_distribution,
    get_useful_comments, get_scenic_comparison
)

# ============================================================
# 创建必要文件夹
# ============================================================
create_dirs()

# ============================================================
# Flask 应用初始化
# ============================================================
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, 'templates'),
    static_folder=os.path.join(PROJECT_ROOT, 'static')
)

app.config['MAX_CONTENT_LENGTH'] = FlaskConfig.MAX_CONTENT_LENGTH  # 32MB
app.config['UPLOAD_FOLDER'] = PathConfig.UPLOAD_DIR
app.config['RESULT_FOLDER'] = PathConfig.RESULT_DIR

# 确保上传和结果目录存在
os.makedirs(PathConfig.UPLOAD_DIR, exist_ok=True)
os.makedirs(PathConfig.RESULT_DIR, exist_ok=True)

# ============================================================
# 设备信息
# ============================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 辅助函数
# ============================================================

def _safe_filename(filename):
    """路径安全：使用 basename 防止路径穿越"""
    return os.path.basename(filename)


# ============================================================
# 游客端路由 —— 页面
# ============================================================

@app.route('/')
def tourist_index():
    """推荐首页"""
    return render_template('tourist/index.html')


@app.route('/scenic/<name>')
def tourist_scenic_detail(name):
    """景区详情页"""
    return render_template('tourist/scenic_detail.html', scenic_name=name)


# ============================================================
# 游客端路由 —— API
# ============================================================

@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    """景区推荐 API
    请求: POST JSON {"text": "需求描述", "top_k": 3}
    响应: JSON {"success": true, "data": [...]}
    """
    try:
        data = request.get_json(force=True)
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': '请提供需求描述（字段: text）'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'success': False, 'error': '需求描述不能为空'}), 400

        top_k = data.get('top_k', 3)
        try:
            top_k = int(top_k)
            top_k = max(1, min(top_k, 12))
        except (ValueError, TypeError):
            top_k = 3

        results = recommend(text, top_k=top_k)
        return jsonify({'success': True, 'data': results})

    except Exception as e:
        print(f"[错误] 推荐失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'推荐失败: {str(e)}'}), 500


@app.route('/api/scenic/detail')
def api_scenic_detail():
    """景区详情 API
    请求: GET ?name=景区名称
    响应: JSON {"success": true, "data": {...}}
    """
    try:
        name = request.args.get('name', '').strip()
        if not name:
            return jsonify({'success': False, 'error': '请提供景区名称（参数: name）'}), 400

        info = get_scenic_info(name)
        if not info or not info.get('address'):
            return jsonify({'success': False, 'error': f'未找到景区: {name}'}), 404

        return jsonify({'success': True, 'data': info})

    except Exception as e:
        print(f"[错误] 获取景区详情失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取详情失败: {str(e)}'}), 500


# ============================================================
# 游客端路由 —— 静态资源代理
# ============================================================

@app.route('/data/scenic_image/<folder>/<filename>')
def serve_scenic_image(folder, filename):
    """提供景区图片"""
    # 路径安全：basename 防止穿越
    safe_folder = _safe_filename(folder)
    safe_filename = _safe_filename(filename)
    image_dir = os.path.join(PathConfig.IMAGES_DIR, safe_folder)
    return send_from_directory(image_dir, safe_filename)


@app.route('/data/video/<filename>')
def serve_video(filename):
    """提供视频文件"""
    safe_filename = _safe_filename(filename)
    return send_from_directory(PathConfig.VIDEO_DIR, safe_filename)


# ============================================================
# 管理端路由 —— 页面
# ============================================================

@app.route('/admin')
def admin_dashboard():
    """管理看板"""
    return render_template('admin/dashboard.html')


@app.route('/admin/analysis')
def admin_analysis():
    """数据分析看板"""
    return render_template('admin/analysis.html')


# ============================================================
# 管理端路由 —— 情感分析 API
# ============================================================

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """单条评论情感分析 API
    请求: POST JSON {"text": "评论内容"}
    响应: JSON {"success": true, "data": {...}, "text": "原文"}
    """
    try:
        data = request.get_json(force=True)
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': '请提供评论内容（字段: text）'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'success': False, 'error': '评论内容不能为空'}), 400
        if len(text) > 2000:
            return jsonify({'success': False, 'error': '评论内容过长，请限制在2000字以内'}), 400

        result = predict_ensemble(text)
        return jsonify({'success': True, 'data': result, 'text': text})

    except Exception as e:
        print(f"[错误] 单条分析失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'分析失败: {str(e)}'}), 500


@app.route('/api/batch', methods=['POST'])
def api_batch():
    """批量CSV分析 API
    请求: multipart/form-data, 字段 "file", CSV需包含评论列
    响应: JSON {"success": true, "data": {"download_url": "...", "stats": {...}}}
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': '请上传CSV文件（字段名: file）'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': '未选择文件'}), 400
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': '仅支持CSV格式文件'}), 400

        # 读取CSV（utf-8-sig 处理 BOM 头）
        content = file.read().decode('utf-8-sig')
        df_input = pd.read_csv(io.StringIO(content))

        # 自动识别评论列（匹配包含"评论"、"内容"、"comment"等的列名）
        comment_col = None
        for col in df_input.columns:
            col_lower = col.lower()
            if '评论' in col or '内容' in col or 'comment' in col_lower or 'review' in col_lower or 'text' in col_lower:
                comment_col = col
                break
        if comment_col is None:
            return jsonify({
                'success': False,
                'error': 'CSV文件中未找到评论内容列',
                'columns_found': list(df_input.columns)
            }), 400

        # 提取评论列表
        comments = df_input[comment_col].fillna('').astype(str).tolist()
        total = len(comments)
        if total == 0:
            return jsonify({'success': False, 'error': 'CSV文件中没有评论数据'}), 400
        if total > 5000:
            return jsonify({
                'success': False,
                'error': f'单次最多处理5000条评论，当前{total}条。请拆分文件后分批上传。'
            }), 400

        print(f"[批量分析] 开始处理 {total} 条评论...")

        # 逐条分析
        results = []
        for i, text in enumerate(comments):
            try:
                if not text.strip():
                    results.append({
                        '评论内容': text, '情感极性': '中性', '置信度': 0.0,
                        '负面概率': 0.0, '中性概率': 1.0, '正面概率': 0.0,
                        '模型一判断': '中性', '模型二判断': '中性',
                    })
                else:
                    r = predict_ensemble(text)
                    results.append({
                        '评论内容': text,
                        '情感极性': r['sentiment'],
                        '置信度': r['confidence'],
                        '负面概率': r['probabilities']['负面'],
                        '中性概率': r['probabilities']['中性'],
                        '正面概率': r['probabilities']['正面'],
                        '模型一判断': r['model1_result']['sentiment'],
                        '模型二判断': r['model2_result']['sentiment'],
                    })
            except Exception:
                results.append({
                    '评论内容': text, '情感极性': '错误', '置信度': 0.0,
                    '负面概率': 0.0, '中性概率': 0.0, '正面概率': 0.0,
                    '模型一判断': '错误', '模型二判断': '错误',
                })
            if (i + 1) % 100 == 0:
                print(f"[批量分析] 进度: {i+1}/{total}")

        # 保存结果CSV（utf-8-sig 支持 Excel 直接打开）
        df_result = pd.DataFrame(results)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_filename = f'batch_result_{timestamp}_{uuid.uuid4().hex[:6]}.csv'
        result_path = os.path.join(PathConfig.RESULT_DIR, result_filename)
        df_result.to_csv(result_path, index=False, encoding='utf-8-sig')

        # 统计各类别数量
        sentiment_counts = df_result['情感极性'].value_counts().to_dict()
        stats = {
            'total': total,
            '正面': sentiment_counts.get('正面', 0),
            '中性': sentiment_counts.get('中性', 0),
            '负面': sentiment_counts.get('负面', 0),
        }
        print(f"[批量分析] 完成! 正面:{stats['正面']} 中性:{stats['中性']} 负面:{stats['负面']}")

        return jsonify({
            'success': True,
            'data': {
                'filename': result_filename,
                'download_url': f'/api/download/{result_filename}',
                'stats': stats,
                'total': total,
            }
        })

    except pd.errors.ParserError:
        return jsonify({'success': False, 'error': 'CSV文件解析失败'}), 400
    except UnicodeDecodeError:
        return jsonify({'success': False, 'error': 'CSV文件编码不支持，请使用UTF-8编码保存'}), 400
    except Exception as e:
        print(f"[错误] 批量分析失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'批量分析失败: {str(e)}'}), 500


@app.route('/api/download/<filename>')
def api_download(filename):
    """下载批量分析结果CSV"""
    try:
        safe_name = _safe_filename(filename)  # 防止路径穿越
        result_path = os.path.join(PathConfig.RESULT_DIR, safe_name)
        if not os.path.exists(result_path):
            return jsonify({'success': False, 'error': '文件不存在或已过期'}), 404
        return send_file(
            result_path, mimetype='text/csv',
            as_attachment=True, download_name=f'情感分析结果_{safe_name}'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': f'下载失败: {str(e)}'}), 500


# ============================================================
# 管理端路由 —— 数据分析 API
# ============================================================

@app.route('/api/admin/stats')
def api_admin_stats():
    """基础统计数据：各景区评论数量"""
    try:
        data = get_scenic_stats()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取统计数据失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取统计数据失败: {str(e)}'}), 500


@app.route('/api/admin/score_dist')
def api_admin_score_dist():
    """评分分布"""
    try:
        data = get_score_distribution()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取评分分布失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取评分分布失败: {str(e)}'}), 500


@app.route('/api/admin/sentiment_ratio')
def api_admin_sentiment_ratio():
    """各景区正负面占比"""
    try:
        data = get_sentiment_ratio()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取情感占比失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取情感占比失败: {str(e)}'}), 500


@app.route('/api/admin/time_trend')
def api_admin_time_trend():
    """评论时间趋势"""
    try:
        data = get_time_trend()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取时间趋势失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取时间趋势失败: {str(e)}'}), 500


@app.route('/api/admin/keywords')
def api_admin_keywords():
    """正面/负面关键词TOP10
    GET参数: scenic=景区名称, sentiment=正面/负面/中性
    """
    try:
        scenic = request.args.get('scenic', '').strip()
        sentiment = request.args.get('sentiment', '正面').strip()
        if not scenic:
            return jsonify({'success': False, 'error': '请提供景区名称（参数: scenic）'}), 400
        if sentiment not in ('正面', '负面', '中性'):
            return jsonify({'success': False, 'error': 'sentiment参数仅支持: 正面/负面/中性'}), 400

        data = get_keywords(scenic, sentiment)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取关键词失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取关键词失败: {str(e)}'}), 500


@app.route('/api/admin/dimensions')
def api_admin_dimensions():
    """景区多维评分
    GET参数: scenic=景区名称
    """
    try:
        scenic = request.args.get('scenic', '').strip()
        if not scenic:
            return jsonify({'success': False, 'error': '请提供景区名称（参数: scenic）'}), 400

        data = get_dimension_scores(scenic)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取多维评分失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取多维评分失败: {str(e)}'}), 500


@app.route('/api/admin/wordcloud')
def api_admin_wordcloud():
    """词云数据
    GET参数: scenic=景区名称
    """
    try:
        scenic = request.args.get('scenic', '').strip()
        if not scenic:
            return jsonify({'success': False, 'error': '请提供景区名称（参数: scenic）'}), 400

        data = get_sentiment_wordcloud_data(scenic)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取词云数据失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取词云数据失败: {str(e)}'}), 500


@app.route('/api/admin/comment_length')
def api_admin_comment_length():
    """评论长度分布"""
    try:
        data = get_comment_length_distribution()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取评论长度分布失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取评论长度分布失败: {str(e)}'}), 500


@app.route('/api/admin/useful_comments')
def api_admin_useful_comments():
    """最有用评论
    GET参数: scenic=景区名称, top_k=5
    """
    try:
        scenic = request.args.get('scenic', '').strip()
        if not scenic:
            return jsonify({'success': False, 'error': '请提供景区名称（参数: scenic）'}), 400

        top_k = request.args.get('top_k', 5)
        try:
            top_k = int(top_k)
            top_k = max(1, min(top_k, 20))
        except (ValueError, TypeError):
            top_k = 5

        data = get_useful_comments(scenic, top_k=top_k)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取有用评论失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取有用评论失败: {str(e)}'}), 500


@app.route('/api/admin/comparison')
def api_admin_comparison():
    """景区综合对比"""
    try:
        data = get_scenic_comparison()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        print(f"[错误] 获取景区对比失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'获取景区对比失败: {str(e)}'}), 500


# ============================================================
# 健康检查 API
# ============================================================

@app.route('/api/health')
def api_health():
    """健康检查接口"""
    return jsonify({
        'success': True,
        'data': {
            'status': 'ok',
            'device': str(device),
            'models': {
                'v1': 'MAFSCN v1 (中性识别敏感)',
                'v2': 'MAFSCN v2 (正负面判断精准)',
                'recommend': 'BERT多分类推荐模型',
            },
            'accuracy': '91.02%',
        }
    })


# ============================================================
# 启动入口
# ============================================================

if __name__ == '__main__':
    import webbrowser

    # 获取本机局域网IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        LAN_IP = s.getsockname()[0]
        s.close()
    except Exception:
        LAN_IP = '本机IP地址'

    print("\n" + "=" * 60)
    print("  MAFSCN 景区智能分析系统")
    print("  双模型融合情感分析 | BERT景区推荐 | 数据可视化")
    print("=" * 60)
    print(f"  本机访问:   http://127.0.0.1:{FlaskConfig.PORT}")
    print(f"  局域网访问: http://{LAN_IP}:{FlaskConfig.PORT}")
    print(f"  使用设备:   {device}")
    print("=" * 60)

    # 预加载模型
    try:
        print("\n[系统] 正在预加载模型...")
        load_models()
        print("[系统] 情感分析模型加载完成")
        load_model()
        print("[系统] 推荐模型加载完成")
        print("[系统] 所有模型加载完成，服务就绪！\n")
    except Exception as e:
        print(f"[系统] 模型预加载失败（将在首次请求时重试）: {e}\n")

    # 启动后自动打开浏览器
    def open_browser():
        url = f'http://127.0.0.1:{FlaskConfig.PORT}'
        print(f"[系统] 正在自动打开浏览器: {url}")
        webbrowser.open(url)

    threading.Timer(1.5, open_browser).start()

    # 启动 Flask 服务
    app.run(
        host=FlaskConfig.HOST,
        port=FlaskConfig.PORT,
        debug=FlaskConfig.DEBUG
    )