"""测试双模型是否产生不同输出"""
import torch
from src.models.mafscn import MAFSCN
from config import PathConfig
from transformers import BertTokenizer

device = torch.device('cpu')

# Load v1
m1 = MAFSCN(bert_model_name=PathConfig.BERT_MODEL_NAME, num_classes=3).to(device)
m1.load_state_dict(torch.load(PathConfig.MODEL_V1_PATH, map_location=device))
m1.eval()

# Load v2
m2 = MAFSCN(bert_model_name=PathConfig.BERT_MODEL_NAME, num_classes=3).to(device)
m2.load_state_dict(torch.load(PathConfig.MODEL_V2_PATH, map_location=device))
m2.eval()

# Compare fusion weights
print('v1 fusion params:', m1.multi_attention.fusion_channel.item(), m1.multi_attention.fusion_spatial.item(), m1.multi_attention.fusion_cross.item())
print('v2 fusion params:', m2.multi_attention.fusion_channel.item(), m2.multi_attention.fusion_spatial.item(), m2.multi_attention.fusion_cross.item())

# Test with multiple sentences
tokenizer = BertTokenizer.from_pretrained(PathConfig.BERT_MODEL_NAME)
test_texts = [
    '故宫非常壮观值得一去',
    '人太多了体验很差',
    '还行吧一般般',
    '风景优美服务态度好',
    '排队两小时玩五分钟太坑了',
]

LABEL_MAP = {0: '负面', 1: '中性', 2: '正面'}
for text in test_texts:
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
    ids = enc['input_ids']
    mask = enc['attention_mask']
    with torch.no_grad():
        logits1, _, _ = m1(ids, mask)
        logits2, _, _ = m2(ids, mask)
        p1 = torch.softmax(logits1, dim=1)[0].cpu().numpy()
        p2 = torch.softmax(logits2, dim=1)[0].cpu().numpy()
    
    label1 = LABEL_MAP[int(p1.argmax())]
    label2 = LABEL_MAP[int(p2.argmax())]
    print(f'\n"{text}"')
    print(f'  v1: {label1} neg={p1[0]:.4f} neu={p1[1]:.4f} pos={p1[2]:.4f}')
    print(f'  v2: {label2} neg={p2[0]:.4f} neu={p2[1]:.4f} pos={p2[2]:.4f}')
    diff = max(abs(p1[i]-p2[i]) for i in range(3))
    print(f'  最大差异: {diff:.4f}')