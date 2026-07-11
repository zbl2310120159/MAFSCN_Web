"""
MAFSCN Web 统一配置文件
所有路径、参数集中管理
"""

import os
import torch


# ============================================================
# 路径配置 —— 全部基于 PROJECT_ROOT 动态获取
# ============================================================
class PathConfig:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # 数据路径
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    DATA_RAW_DIR = os.path.join(DATA_DIR, 'raw')
    DATA_PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
    DATA_PREPROCESSED = os.path.join(DATA_PROCESSED_DIR, 'preprocessed_data.csv')
    SCENIC_INFO = os.path.join(DATA_DIR, 'scenic_info.xlsx')

    # 资源路径
    ASSETS_DIR = os.path.join(DATA_DIR, 'assets')
    IMAGES_DIR = os.path.join(ASSETS_DIR, 'images')
    VIDEO_DIR = os.path.join(ASSETS_DIR, 'video')

    # 模型路径
    MODEL_DIR = os.path.join(PROJECT_ROOT, 'results', 'models')
    MODEL_V1_PATH = os.path.join(MODEL_DIR, 'best_mafscn_model_v1.pth')
    MODEL_V2_PATH = os.path.join(MODEL_DIR, 'best_mafscn_model_fixed.pth')
    RECOMMEND_MODEL_PATH = os.path.join(MODEL_DIR, 'bert_recommend_model.pth')
    KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, 'kmeans_model.pkl')
    SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')
    LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

    # BERT模型路径
    BERT_MODEL_NAME = os.path.join(PROJECT_ROOT, 'bert-base-chinese')

    # 结果输出路径
    FIGURE_DIR = os.path.join(PROJECT_ROOT, 'results', 'figures')

    # 上传/下载路径
    UPLOAD_DIR = os.path.join(PROJECT_ROOT, 'uploads')
    RESULT_DIR = os.path.join(PROJECT_ROOT, 'results', 'batch_results')


# ============================================================
# 数据配置
# ============================================================
class DataConfig:
    MAX_SEQ_LEN = 128       # 最大序列长度
    TEST_SIZE = 0.3         # 测试集比例
    RANDOM_SEED = 42        # 随机种子

    # 情感标签映射：1-2→负面(0), 3→中性(1), 4-5→正面(2)
    SENTIMENT_MAP = {
        1: 0,  # 负面
        2: 0,  # 负面
        3: 1,  # 中性
        4: 2,  # 正面
        5: 2   # 正面
    }

    LABEL_NAMES = {0: '负面', 1: '中性', 2: '正面'}
    LABEL_COLORS = {'负面': '#ef4444', '中性': '#9ca3af', '正面': '#22c55e'}
    LABEL_ICONS = {'负面': '😞', '中性': '😐', '正面': '😊'}

    # 保留的核心列
    KEEP_COLUMNS = [
        '评论ID', '用户昵称', '综合评分', '评论内容', '发布时间',
        '有用数', '回复数', '图片数量', 'IP归属地', '是否有视频'
    ]

    # 景区名称 → 图片文件夹映射
    SCENIC_IMAGE_MAP = {
        '冰雪大世界': '冰雪大世界',
        '兵马俑': '兵马俑',
        '都江堰': '都江堰',
        '峨眉山': '峨眉山',
        '故宫': '故宫',
        '龙门石窟': '龙门石窟',
        '清明上河园': '清明上河园',
        '上海迪士尼': '上海迪士尼',
        '万岁山武侠城': '万岁山武侠城',
        '玉龙雪山': '玉龙雪山',
        '云台山': '云台山',
        '长白山': '长白山',
    }


# ============================================================
# 模型配置
# ============================================================
class ModelConfig:
    HIDDEN_SIZE = 768       # BERT隐藏层维度
    NUM_CLASSES = 3         # 三分类：负面/中性/正面
    DROPOUT = 0.3           # Dropout率
    REDUCTION = 16          # 通道注意力压缩比

    # 双模型融合权重
    ENSEMBLE_WEIGHTS = {
        'negative': (0.3, 0.7),   # 负面: v1权重0.3, v2权重0.7
        'neutral':  (0.8, 0.2),   # 中性: v1权重0.8, v2权重0.2
        'positive': (0.3, 0.7),   # 正面: v1权重0.3, v2权重0.7
    }
    NEUTRAL_THRESHOLD = 0.3  # 中性概率≥30%直接判中性


# ============================================================
# 训练配置
# ============================================================
class TrainConfig:
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE_BERT = 2e-5
    LEARNING_RATE_FUSION = 5e-3
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0
    EARLY_STOP_PATIENCE = 3
    TRAIN_TEST_SPLIT = 0.2       # 训练/测试划分比例
    RANDOM_SEED = 42


# ============================================================
# Flask配置
# ============================================================
class FlaskConfig:
    HOST = '0.0.0.0'
    PORT = 5000
    DEBUG = False
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 上传限制32MB


# ============================================================
# 设备配置
# ============================================================
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# 创建必要文件夹
# ============================================================
def create_dirs():
    """创建所有需要的文件夹"""
    dirs = [
        PathConfig.DATA_PROCESSED_DIR,
        PathConfig.MODEL_DIR,
        PathConfig.FIGURE_DIR,
        PathConfig.UPLOAD_DIR,
        os.path.join(PathConfig.PROJECT_ROOT, 'results', 'batch_results'),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✅ 所有文件夹已创建")


# ============================================================
# 随机种子
# ============================================================
def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    create_dirs()
    print(f"项目根目录: {PathConfig.PROJECT_ROOT}")
    print(f"设备: {get_device()}")
    print(f"BERT路径: {PathConfig.BERT_MODEL_NAME}")
    print(f"预处理数据: {PathConfig.DATA_PREPROCESSED}")