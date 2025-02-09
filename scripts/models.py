# models.py
import torch
import torch.nn as nn

# 共享特征提取模块（例如使用 Transformer 或 CNN）
class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        # 伪代码：构建编码器网络
        self.encoder = nn.Sequential(
            # nn.Conv1d(...), nn.ReLU(), nn.BatchNorm1d(...), ...
        )

    def forward(self, x):
        features = self.encoder(x)
        return features

# 序列级分支：预测整体结合概率（例如使用全连接层）
class SeqBranch(nn.Module):
    def __init__(self, config):
        super(SeqBranch, self).__init__()
        self.fc = nn.Linear(in_features=128, out_features=1)  # in_features 根据 encoder 输出设定

    def forward(self, features):
        # 输出结合概率（用 sigmoid 激活）
        out = torch.sigmoid(self.fc(features))
        return out

# 残基级分支：通过交叉注意力和卷积模块预测残基间接触/距离矩阵
class ResBranch(nn.Module):
    def __init__(self, config):
        super(ResBranch, self).__init__()
        # 伪代码：构建交叉注意力模块及卷积层
        self.cross_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, features_seq, features_epitope):
        # features_seq 与 features_epitope 均为张量
        # 通过交叉注意力模块计算注意力矩阵
        attn_output, _ = self.cross_attention(features_seq, features_epitope, features_epitope)
        # 将注意力矩阵调整为 2D 格式后输入卷积层
        attn_matrix = attn_output.unsqueeze(1)  # 假设需要添加 channel 维度
        res_output = self.cnn(attn_matrix)
        return res_output

# 结构融合分支：融合结构信息进行预测（例如利用图卷积网络）
class StructBranch(nn.Module):
    def __init__(self, config):
        super(StructBranch, self).__init__()
        # 伪代码：定义结构信息编码模块，如 GCN
        self.gcn = nn.Sequential(
            # 例如：GraphConv(...), ReLU(), GraphConv(...), ...
        )
        self.fc = nn.Linear(in_features=64, out_features=1)

    def forward(self, features, structure_data):
        # structure_data 为结构相关的输入数据
        struct_features = self.gcn(structure_data)
        # 融合 sequence 特征与结构特征，方式可选：拼接、加权求和等
        fused = torch.cat([features, struct_features], dim=-1)
        out = self.fc(fused)
        return out

# 整体多任务模型
class MultiTaskModel(nn.Module):
    def __init__(self, config):
        super(MultiTaskModel, self).__init__()
        self.feature_extractor = FeatureExtractor(config)
        self.seq_branch = SeqBranch(config)
        self.res_branch = ResBranch(config)
        self.struct_branch = StructBranch(config)

    def forward(self, x_seq, x_epitope, structure_data=None):
        # 先提取 TCR 序列的通用特征
        features = self.feature_extractor(x_seq)
        # 序列级预测
        seq_pred = self.seq_branch(features)
        # 残基级预测：假设 epitope 也经过相同编码（或独立编码）
        res_pred = self.res_branch(features, x_epitope)
        # 结构分支：若提供结构数据，则进行融合预测
        if structure_data is not None:
            struct_pred = self.struct_branch(features, structure_data)
        else:
            struct_pred = None
        return seq_pred, res_pred, struct_pred
