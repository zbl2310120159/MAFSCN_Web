"""
多注意力融合模块
包含：通道注意力、空间注意力、交叉注意力、自适应融合

论文参考：MAFSCN - Multi-Attention Fusion Sentiment Classification Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    通道注意力模块 (论文3.3.2)
    计算C×C通道相关性矩阵，捕捉通道间依赖关系
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.channels = channels

        # 降维后再升维，减少计算量
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, hidden_size] - CLS token特征
        Returns:
            weighted: [batch_size, hidden_size] - 加权后的特征
        """
        weights = self.fc(x)  # [batch_size, channels]
        return x * weights


class SpatialAttention(nn.Module):
    """
    空间注意力模块 (论文3.3.3)
    关注文本中的关键位置（情感词、转折词等）
    """

    def __init__(self, hidden_size=768):
        super(SpatialAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_size] - 完整序列特征
        Returns:
            weighted: [batch_size, seq_len, hidden_size] - 加权后的特征
        """
        weights = self.attention(x)  # [batch_size, seq_len, 1]
        return x * weights


class CrossAttention(nn.Module):
    """
    交叉注意力模块 (论文3.3.4)
    捕捉不同位置特征之间的交互关系，缓解长距离依赖
    """

    def __init__(self, hidden_size=768):
        super(CrossAttention, self).__init__()
        self.hidden_size = hidden_size

        # Q, K, V 投影
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, hidden_size] - 完整序列特征
        Returns:
            output: [batch_size, seq_len, hidden_size] - 加权后的特征
            attention_weights: [batch_size, seq_len, seq_len] - 注意力权重
        """
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        # 残差连接
        return output + x, attention_weights


class MultiAttentionFusion(nn.Module):
    """
    多注意力融合模块 (论文3.3.5)
    组合通道、空间、交叉注意力，自适应学习融合权重
    """

    def __init__(self, hidden_size=768, reduction=16):
        super(MultiAttentionFusion, self).__init__()

        self.channel_attn = ChannelAttention(hidden_size, reduction)
        self.spatial_attn = SpatialAttention(hidden_size)
        self.cross_attn = CrossAttention(hidden_size)

        # 自适应融合权重（可学习参数）
        self.fusion_channel = nn.Parameter(torch.tensor(0.5))
        self.fusion_spatial = nn.Parameter(torch.tensor(0.3))
        self.fusion_cross = nn.Parameter(torch.tensor(0.2))

    def forward(self, cls_embedding, sequence_output):
        """
        Args:
            cls_embedding: [batch, hidden] - BERT的[CLS] token输出
            sequence_output: [batch, seq_len, hidden] - BERT完整序列输出

        Returns:
            fused_features: [batch, hidden] - 融合后的特征
            weights_dict: dict - 各注意力的权重
            cross_weights: [batch, num_heads, seq_len, seq_len] - 交叉注意力权重
        """
        # 1. 通道注意力（作用于CLS token）
        channel_out = self.channel_attn(cls_embedding)

        # 2. 空间注意力（作用于完整序列，然后聚合）
        spatial_out = self.spatial_attn(sequence_output)
        spatial_out = spatial_out.mean(dim=1)  # [batch, hidden]

        # 3. 交叉注意力（作用于完整序列，然后聚合）
        cross_out, cross_weights = self.cross_attn(sequence_output)
        cross_out = cross_out.mean(dim=1)  # [batch, hidden]

        # 4. 自适应融合
        fusion_logits = torch.stack([self.fusion_channel, self.fusion_spatial, self.fusion_cross])
        fusion_weights = F.softmax(fusion_logits, dim=0)

        fused = (fusion_weights[0] * channel_out +
                 fusion_weights[1] * spatial_out +
                 fusion_weights[2] * cross_out)

        weights_dict = {
            'channel': fusion_weights[0].item(),
            'spatial': fusion_weights[1].item(),
            'cross': fusion_weights[2].item()
        }

        return fused, weights_dict, cross_weights