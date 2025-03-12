import torch
import torch.nn as nn
from encode_sequences import encode_sequences  # 导入函数
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import sys
import os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEIM_CONFIG, TCR_MODEL_NAME, PROT_MODEL_NAME, DEVICE

# 定义残差网络模块，用于构建交互层中的残差块
class ResNet(nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn

    def forward(self, data):
        tmp_data = self.cnn(data)
        out = tmp_data + data
        return out

# 旋转位置编码实现
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # 确保最后一个维度是偶数
        if x.shape[-1] % 2 != 0:
            # 如果不是偶数，添加一个零列使其成为偶数
            pad = torch.zeros(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        
        seq_len = x.shape[2]
        # 确保dim是偶数
        dim = self.dim if self.dim % 2 == 0 else self.dim - 1
        
        # 生成频率
        freqs = 10000 ** (-2 * torch.arange(0, dim // 2, device=x.device, dtype=torch.float32) / dim)
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.einsum('i,j->ij', position, freqs)
        
        # 确保freqs的形状与x兼容
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # 使用更安全的方式重塑张量
        try:
            # 尝试使用view_as_complex
            x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
            x_complex = torch.view_as_complex(x_reshaped)
            emb_complex = torch.polar(torch.ones_like(x_complex), emb.to(x.device))
            x_rotated = torch.view_as_real(x_complex * emb_complex).flatten(-2)
        except RuntimeError as e:
            # 如果失败，使用替代方法
            x_rotated = x  # 暂时不应用旋转，只返回原始输入
        
        return x_rotated.type_as(x)

# 定义 TEIM 模型类
class TEIM(nn.Module):
    def __init__(self, file_path=None):
        super().__init__()
        # 从配置文件中获取参数
        self.dim_hidden = TEIM_CONFIG["dim_hidden"]
        self.layers_inter = TEIM_CONFIG["layers_inter"]
        self.dim_seqlevel = TEIM_CONFIG["dim_seqlevel"]
        self.dropout_rate = TEIM_CONFIG.get("dropout_rate", 0.2)  # 使用get方法，如果不存在则使用默认值
        
        # 分别定义 CDR3 和 Epitope 的嵌入维度
        self.dim_emb_cdr3 = TEIM_CONFIG.get("dim_emb_cdr3", 768)  # 使用配置中的值或默认值
        self.dim_emb_epi = TEIM_CONFIG.get("dim_emb_epi", 1024)   # 使用配置中的值或默认值
        
        # 初始化tokenizer和模型
        self.tcr_tokenizer = AutoTokenizer.from_pretrained(TCR_MODEL_NAME)
        self.tcr_model = AutoModel.from_pretrained(TCR_MODEL_NAME)

        self.prot_tokenizer = AutoTokenizer.from_pretrained(PROT_MODEL_NAME)
        self.prot_model = AutoModel.from_pretrained(PROT_MODEL_NAME)

        # 定义 CDR3 序列的旋转位置编码模块
        self.rope_cdr3 = RotaryPositionEmbedding(self.dim_emb_cdr3)
        # 定义抗原表位序列的旋转位置编码模块
        self.rope_epi = RotaryPositionEmbedding(self.dim_emb_epi)

        # 定义 CDR3 序列的特征提取模块
        self.seq_cdr3 = nn.Sequential(
            nn.Conv1d(self.dim_emb_cdr3, self.dim_hidden, 1),
            nn.BatchNorm1d(self.dim_hidden),
            nn.ReLU(),
        )
        # 定义抗原表位序列的特征提取模块
        self.seq_epi = nn.Sequential(
            nn.Conv1d(self.dim_emb_epi, self.dim_hidden, 1),
            nn.BatchNorm1d(self.dim_hidden),
            nn.ReLU(),
        )

        self.inter_layers = nn.ModuleList([
            nn.Sequential(
                ResNet(nn.Conv2d(self.dim_hidden, self.dim_hidden, kernel_size=3, padding=1)),
                nn.BatchNorm2d(self.dim_hidden),
                nn.ReLU(),
            ) for _ in range(self.layers_inter)
        ])

        self.seqlevel_outlyer = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Dropout(self.dropout_rate),  # 使用配置中的dropout_rate
            nn.Linear(self.dim_hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, cdr3_sequences, epitope_sequences):
        """
        前向传播方法，接受CDR3和Epitope序列作为输入
        
        参数:
        cdr3_sequences: 批次的CDR3序列
        epitope_sequences: 批次的Epitope序列
        
        返回:
        包含序列级预测和交互映射的字典
        """
        device = next(self.parameters()).device  # 获取模型所在设备
        
        # 编码CDR3序列
        batch_size = len(cdr3_sequences)
        cdr3_embeddings = []
        for cdr3_seq in cdr3_sequences:
            inputs = self.tcr_tokenizer(cdr3_seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = self.tcr_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            cdr3_embeddings.append(cls_embedding)
        
        # 编码Epitope序列
        epitope_embeddings = []
        for epitope_seq in epitope_sequences:
            inputs = self.prot_tokenizer(epitope_seq, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = self.prot_model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            epitope_embeddings.append(cls_embedding)
        
        # 将嵌入向量堆叠成批次
        cdr3_emb = torch.cat(cdr3_embeddings, dim=0)
        epi_emb = torch.cat(epitope_embeddings, dim=0)

        # 调整 cdr3_emb 的维度
        cdr3_emb = cdr3_emb.unsqueeze(1)
        cdr3_emb = cdr3_emb.transpose(1, 2)

        # 调整 epi_emb 的维度
        epi_emb = epi_emb.unsqueeze(1)
        epi_emb = epi_emb.transpose(1, 2)

        # 应用旋转位置编码
        try:
            cdr3_emb_RoPE = self.rope_cdr3(cdr3_emb)
            epi_emb_RoPE = self.rope_epi(epi_emb)
        except Exception:
            # 如果RoPE失败，直接使用原始嵌入
            cdr3_emb_RoPE = cdr3_emb
            epi_emb_RoPE = epi_emb

        ## sequence features
        cdr3_feat = self.seq_cdr3(cdr3_emb_RoPE)  # batch_size, dim_hidden, seq_len
        epi_feat = self.seq_epi(epi_emb_RoPE)

        len_cdr3 = cdr3_feat.shape[2]
        len_epi = epi_feat.shape[2]

        ## get init inter map
        cdr3_feat_mat = cdr3_feat.unsqueeze(3).repeat([1, 1, 1, len_epi])
        epi_feat_mat = epi_feat.unsqueeze(2).repeat([1, 1, len_cdr3, 1])

        inter_map = cdr3_feat_mat * epi_feat_mat

        ## inter layers features
        for layer in self.inter_layers:
            inter_map = layer(inter_map)

        ## output layers
        seqlevel_out = self.seqlevel_outlyer(inter_map)

        return {
            'seqlevel_out': seqlevel_out,
            'inter_map': inter_map,
        }
