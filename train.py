"""
MAFSCN 训练脚本
包含：MAFSCN情感分析模型训练 + BERT推荐模型训练
参考 MAFSCN_Project/src/train.py 和 train_recommend.py 适配

用法:
  python train.py sentiment          - 训练MAFSCN情感分析模型
  python train.py recommend          - 训练BERT推荐模型
  python train.py all                - 训练所有模型
  python train.py --epochs 5         - 指定训练轮数
  python train.py --batch-size 16    - 指定批量大小
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
import joblib

warnings.filterwarnings('ignore')

from config import PathConfig, TrainConfig, DataConfig, set_seed

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============================================================
# 通用训练工具
# ============================================================

def save_checkpoint(model, optimizer, scheduler, epoch, best_metric, filepath):
    """保存检查点"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_metric': best_metric,
    }, filepath)
    print(f"  ✅ 检查点已保存: {filepath}")


def load_checkpoint(filepath, model, optimizer, scheduler):
    """加载检查点"""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
    print(f"  ✅ 加载检查点: epoch {epoch}, best_metric={best_metric:.4f}")
    return epoch, best_metric


def save_log(log_data, filepath):
    """保存训练日志"""
    df = pd.DataFrame(log_data)
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"  ✅ 日志已保存: {filepath}")


# ============================================================
# MAFSCN 情感分析训练
# ============================================================

def train_sentiment_epoch(model, loader, optimizer, scheduler, criterion, device):
    """MAFSCN训练一个epoch"""
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="训练"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits, _, _ = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainConfig.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_sentiment(model, loader, device):
    """评估MAFSCN模型"""
    model.eval()
    all_preds = []
    all_labels = []
    fusion_weights_collect = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="评估"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits, attn_weights, _ = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            fusion_weights_collect.append([
                attn_weights['channel'],
                attn_weights['spatial'],
                attn_weights['cross']
            ])

    return all_preds, all_labels, np.mean(fusion_weights_collect, axis=0)


def plot_fusion_weights_history(history, save_path):
    """绘制融合权重随训练轮次的变化曲线"""
    history = np.array(history)
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history) + 1)

    plt.plot(epochs, history[:, 0], 'o-', label='通道注意力', color='#FF6B6B', linewidth=2, markersize=8)
    plt.plot(epochs, history[:, 1], 's-', label='空间注意力', color='#4ECDC4', linewidth=2, markersize=8)
    plt.plot(epochs, history[:, 2], '^-', label='交叉注意力', color='#45B7D1', linewidth=2, markersize=8)

    plt.axhline(y=0.333, color='gray', linestyle='--', alpha=0.5, label='均匀分布线')
    plt.xlabel('训练轮次 (Epoch)', fontsize=12)
    plt.ylabel('融合权重', fontsize=12)
    plt.title('多注意力融合权重随训练变化曲线', fontsize=14)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train_mafscn(args):
    """训练MAFSCN情感分析模型"""
    from src.models.mafscn import MAFSCN
    from src.data.dataset import SentimentDataset, load_sentiment_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print("  MAFSCN情感分析模型训练")
    print(f"{'=' * 60}")
    print(f"  设备: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 配置
    BERT_PATH = PathConfig.BERT_MODEL_NAME
    BEST_MODEL_PATH = PathConfig.MODEL_V2_PATH
    CHECKPOINT_PATH = os.path.join(PathConfig.MODEL_DIR, 'checkpoint_mafscn.pt')
    LOG_PATH = os.path.join(PathConfig.MODEL_DIR, 'training_log_mafscn.csv')
    ATTENTION_DIR = os.path.join(PathConfig.FIGURE_DIR, 'attention')
    os.makedirs(ATTENTION_DIR, exist_ok=True)

    NUM_EPOCHS = args.epochs or TrainConfig.EPOCHS
    BATCH_SIZE = args.batch_size or TrainConfig.BATCH_SIZE
    LR_BERT = TrainConfig.LEARNING_RATE_BERT
    LR_FUSION = TrainConfig.LEARNING_RATE_FUSION

    # 1. 加载数据
    print(f"\n[1/6] 加载预处理数据...")
    texts, labels, df = load_sentiment_data()

    # 2. 划分数据集
    print(f"\n[2/6] 划分数据集 (8:2)...")
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=DataConfig.RANDOM_SEED, stratify=labels
    )
    print(f"  训练集: {len(X_train)}条")
    print(f"  测试集: {len(X_test)}条")

    # 3. 创建数据加载器
    print(f"\n[3/6] 创建数据加载器...")
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    train_dataset = SentimentDataset(X_train, y_train, tokenizer)
    test_dataset = SentimentDataset(X_test, y_test, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. 创建模型
    print(f"\n[4/6] 创建MAFSCN模型...")
    model = MAFSCN(bert_model_name=BERT_PATH, num_classes=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # 打印初始融合权重
    with torch.no_grad():
        fusion_logits = torch.stack([
            model.multi_attention.fusion_channel,
            model.multi_attention.fusion_spatial,
            model.multi_attention.fusion_cross
        ])
        init_weights = torch.softmax(fusion_logits, dim=0)
        print(f"  初始融合权重: 通道={init_weights[0].item():.3f}, "
              f"空间={init_weights[1].item():.3f}, 交叉={init_weights[2].item():.3f}")

    # 5. 优化器
    print(f"\n[5/6] 设置优化器...")
    fusion_params = [
        model.multi_attention.fusion_channel,
        model.multi_attention.fusion_spatial,
        model.multi_attention.fusion_cross
    ]
    other_params = [p for n, p in model.named_parameters()
                    if not any(fp is p for fp in fusion_params)]

    optimizer = AdamW([
        {'params': fusion_params, 'lr': LR_FUSION},
        {'params': other_params, 'lr': LR_BERT}
    ])

    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * TrainConfig.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    # 6. 检查断点续训
    start_epoch = 0
    best_f1 = 0
    fusion_weight_history = []
    train_log = []

    if os.path.exists(CHECKPOINT_PATH) and args.resume:
        start_epoch, best_f1, _ = load_checkpoint(CHECKPOINT_PATH, model, optimizer, scheduler)
        if os.path.exists(LOG_PATH):
            train_log = pd.read_csv(LOG_PATH).to_dict('records')

    # 7. 训练
    print(f"\n[6/6] 开始训练 ({NUM_EPOCHS}轮)...")
    print(f"{'=' * 60}")

    patience_counter = 0

    for epoch in range(start_epoch, NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'─' * 40}")

        train_loss = train_sentiment_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        preds, labels_true, avg_weights = evaluate_sentiment(model, test_loader, device)
        acc = accuracy_score(labels_true, preds)
        f1 = f1_score(labels_true, preds, average='weighted')

        fusion_weight_history.append(avg_weights)

        log_entry = {
            'epoch': epoch + 1, 'train_loss': train_loss,
            'accuracy': acc, 'f1_score': f1,
            'channel_weight': avg_weights[0],
            'spatial_weight': avg_weights[1],
            'cross_weight': avg_weights[2],
        }
        train_log.append(log_entry)

        print(f"  训练损失: {train_loss:.4f}")
        print(f"  测试准确率: {acc:.4f}")
        print(f"  测试F1: {f1:.4f}")
        print(f"  融合权重: 通道={avg_weights[0]:.3f}, 空间={avg_weights[1]:.3f}, 交叉={avg_weights[2]:.3f}")

        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✅ 保存最佳模型 (F1={f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        # 保存检查点
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_f1, CHECKPOINT_PATH)
        save_log(train_log, LOG_PATH)

        # 绘制权重变化图
        plot_fusion_weights_history(fusion_weight_history,
                                    os.path.join(ATTENTION_DIR, 'fusion_weights_history.png'))

        # 早停
        if patience_counter >= TrainConfig.EARLY_STOP_PATIENCE:
            print(f"\n  ⚠️ 早停: {TrainConfig.EARLY_STOP_PATIENCE}轮无改善")
            break

    # 最终评估
    print(f"\n{'=' * 60}")
    print("  最终评估结果")
    print(f"{'=' * 60}")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    preds, labels_true, final_weights = evaluate_sentiment(model, test_loader, device)

    accuracy = accuracy_score(labels_true, preds)
    f1 = f1_score(labels_true, preds, average='weighted')

    print(f"\n  准确率: {accuracy:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  融合权重: 通道={final_weights[0]:.4f}, 空间={final_weights[1]:.4f}, 交叉={final_weights[2]:.4f}")
    print(f"\n  分类报告:")
    print(classification_report(labels_true, preds, target_names=['负面', '中性', '正面']))

    print(f"\n  模型保存: {BEST_MODEL_PATH}")
    print(f"  日志保存: {LOG_PATH}")


# ============================================================
# BERT 推荐模型训练
# ============================================================

def train_recommend_epoch(model, loader, optimizer, scheduler, criterion, device):
    """BERT推荐模型训练一个epoch"""
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="训练"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainConfig.MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_recommend(model, loader, device):
    """评估BERT推荐模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="评估"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def train_bert_recommend(args):
    """训练BERT推荐模型"""
    from src.models.bert_classifier import BERTMultiClassifier
    from src.data.dataset import RecommendDataset, load_recommend_data

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print("  BERT景区推荐模型训练")
    print(f"{'=' * 60}")
    print(f"  设备: {device}")

    BERT_PATH = PathConfig.BERT_MODEL_NAME
    BEST_MODEL_PATH = PathConfig.RECOMMEND_MODEL_PATH
    LOG_PATH = os.path.join(PathConfig.MODEL_DIR, 'training_log_recommend.csv')

    NUM_EPOCHS = args.epochs or TrainConfig.EPOCHS
    BATCH_SIZE = args.batch_size or TrainConfig.BATCH_SIZE
    LEARNING_RATE = TrainConfig.LEARNING_RATE_BERT

    # 1. 加载数据
    print(f"\n[1/7] 加载数据...")
    texts, scenic_names, labels, label_encoder, df = load_recommend_data()

    # 2. 数据清洗：过滤短文本
    print(f"\n[2/7] 数据清洗...")
    min_words = 5
    filtered_texts, filtered_labels = [], []
    for text, label in zip(texts, labels):
        if len(str(text).split()) >= min_words:
            filtered_texts.append(text)
            filtered_labels.append(label)
    print(f"  过滤短评论(<{min_words}词): {len(texts)} → {len(filtered_texts)}条")
    texts, labels = filtered_texts, filtered_labels

    # 3. 划分数据集
    print(f"\n[3/7] 划分数据集 (7:1:2)...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=DataConfig.RANDOM_SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, random_state=DataConfig.RANDOM_SEED, stratify=y_temp
    )
    print(f"  训练集: {len(X_train)}条")
    print(f"  验证集: {len(X_val)}条")
    print(f"  测试集: {len(X_test)}条")

    # 4. 创建数据加载器
    print(f"\n[4/7] 创建数据加载器...")
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    train_dataset = RecommendDataset(X_train, y_train, tokenizer)
    val_dataset = RecommendDataset(X_val, y_val, tokenizer)
    test_dataset = RecommendDataset(X_test, y_test, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. 创建模型
    print(f"\n[5/7] 创建BERT推荐模型...")
    model = BERTMultiClassifier(len(scenic_names), BERT_PATH).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")
    print(f"  景区数: {len(scenic_names)}")

    # 6. 训练
    print(f"\n[6/7] 开始训练 ({NUM_EPOCHS}轮)...")
    print(f"{'=' * 60}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * TrainConfig.WARMUP_RATIO),
        num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    patience_counter = 0
    train_log = []

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'─' * 40}")

        train_loss = train_recommend_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        val_preds, val_labels = evaluate_recommend(model, val_loader, device)
        val_acc = accuracy_score(val_labels, val_preds)

        log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_acc': val_acc}
        train_log.append(log_entry)

        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证准确率: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  ✅ 保存最佳模型 (准确率={val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        save_log(train_log, LOG_PATH)

        if patience_counter >= TrainConfig.EARLY_STOP_PATIENCE:
            print(f"\n  ⚠️ 早停: {TrainConfig.EARLY_STOP_PATIENCE}轮无改善")
            break

    # 7. 测试集评估
    print(f"\n[7/7] 测试集评估...")
    print(f"{'=' * 60}")

    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    test_preds, test_labels = evaluate_recommend(model, test_loader, device)
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"\n  测试集准确率: {test_acc:.4f}")
    print(f"\n  分类报告:")
    print(classification_report(test_labels, test_preds, target_names=scenic_names))

    # 保存LabelEncoder
    joblib.dump(label_encoder, PathConfig.LABEL_ENCODER_PATH)
    print(f"\n  模型保存: {BEST_MODEL_PATH}")
    print(f"  编码器保存: {PathConfig.LABEL_ENCODER_PATH}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='MAFSCN 训练脚本')
    parser.add_argument('model', choices=['sentiment', 'recommend', 'all'],
                        help='训练模型: sentiment/recommend/all')
    parser.add_argument('--epochs', '-e', type=int, help='训练轮数(默认使用config配置)')
    parser.add_argument('--batch-size', '-b', type=int, help='批量大小(默认使用config配置)')
    parser.add_argument('--resume', '-r', action='store_true', help='从检查点恢复训练')
    parser.add_argument('--seed', '-s', type=int, default=42, help='随机种子(默认42)')

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(PathConfig.MODEL_DIR, exist_ok=True)

    if args.model in ('sentiment', 'all'):
        train_mafscn(args)

    if args.model in ('recommend', 'all'):
        train_bert_recommend(args)

    print(f"\n{'=' * 60}")
    print("  训练完成！")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()