"""
验证模型权重加载
测试新模型代码能否正确加载旧权重
"""

import os
import sys
import torch

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import PathConfig, get_device

print("=" * 60)
print("模型权重加载验证")
print("=" * 60)

device = get_device()
print(f"设备: {device}")

# 设置HF镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

results = {}

# -------------------------------------------------------
# 1. 验证 MAFSCN v1 模型
# -------------------------------------------------------
print("\n[1/3] 验证 MAFSCN v1 模型...")
try:
    from src.models.mafscn import MAFSCN
    model_v1 = MAFSCN(
        bert_model_name=PathConfig.BERT_MODEL_NAME,
        num_classes=3,
        dropout=0.3,
        use_multi_layer=False
    ).to(device)

    state_dict = torch.load(PathConfig.MODEL_V1_PATH, map_location=device, weights_only=False)
    model_v1.load_state_dict(state_dict, strict=True)
    model_v1.eval()
    print("  ✅ MAFSCN v1 加载成功 (strict=True)")
    results['v1'] = '✅'
    del model_v1, state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    print(f"  ❌ MAFSCN v1 加载失败: {e}")
    results['v1'] = f'❌ {e}'

# -------------------------------------------------------
# 2. 验证 MAFSCN v2 (fixed) 模型
# -------------------------------------------------------
print("\n[2/3] 验证 MAFSCN v2 (fixed) 模型...")
try:
    from src.models.mafscn import MAFSCN
    model_v2 = MAFSCN(
        bert_model_name=PathConfig.BERT_MODEL_NAME,
        num_classes=3,
        dropout=0.3,
        use_multi_layer=False
    ).to(device)

    state_dict = torch.load(PathConfig.MODEL_V2_PATH, map_location=device, weights_only=False)
    model_v2.load_state_dict(state_dict, strict=True)
    model_v2.eval()
    print("  ✅ MAFSCN v2 (fixed) 加载成功 (strict=True)")
    results['v2'] = '✅'
    del model_v2, state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    print(f"  ❌ MAFSCN v2 (fixed) 加载失败: {e}")
    results['v2'] = f'❌ {e}'

# -------------------------------------------------------
# 3. 验证 BERT 推荐模型
# -------------------------------------------------------
print("\n[3/3] 验证 BERT 推荐模型...")
try:
    from src.models.bert_classifier import BERTMultiClassifier

    # 需要知道景区数量来确定num_classes
    import pandas as pd
    df = pd.read_csv(PathConfig.DATA_PREPROCESSED)
    num_scenic = df['景区'].nunique()
    print(f"  景区数量: {num_scenic}")

    model_rec = BERTMultiClassifier(
        num_classes=num_scenic,
        bert_model_name=PathConfig.BERT_MODEL_NAME,
        dropout=0.3
    ).to(device)

    state_dict = torch.load(PathConfig.RECOMMEND_MODEL_PATH, map_location=device, weights_only=False)
    model_rec.load_state_dict(state_dict, strict=True)
    model_rec.eval()
    print("  ✅ BERT 推荐模型加载成功 (strict=True)")
    results['recommend'] = '✅'
    del model_rec, state_dict
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
except Exception as e:
    print(f"  ❌ BERT 推荐模型加载失败: {e}")
    results['recommend'] = f'❌ {e}'

# -------------------------------------------------------
# 汇总
# -------------------------------------------------------
print("\n" + "=" * 60)
print("验证结果汇总:")
for name, status in results.items():
    print(f"  {name}: {status}")

all_ok = all(v == '✅' for v in results.values())
print(f"\n{'🎉 全部模型验证通过！' if all_ok else '⚠️ 有模型验证失败，请检查'}")
print("=" * 60)